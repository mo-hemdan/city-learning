"""
infer.py  –  Impute missing road attributes using a trained MultiAttrGAT checkpoint.

Usage
-----
    python infer.py \
        --city jakarta \
        --data_dir ./data/raw_data \
        --checkpoint_dir ./checkpoints \
        --output ./data/raw_data/jakarta_imputed.parquet
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.expanduser("~/websites/mapedia"))

import numpy as np
import pandas as pd
import geopandas as gpd
import torch

_CONSTRAINTS_PATH = os.path.join(os.path.dirname(__file__), "speed_constraints.json")
with open(_CONSTRAINTS_PATH) as _f:
    SPEED_CONSTRAINTS = json.load(_f)


def apply_speed_constraints(arr, city, attr):
    """Round to nearest 5 and clip to city-specific [min, max]."""
    c = SPEED_CONSTRAINTS.get(city, {}).get(attr, {})
    lo = c.get("min", 0)
    hi = c.get("max", 999)
    rounded = np.round(arr / 5.0) * 5.0
    return np.clip(rounded, lo, hi).astype(np.float32)

from src.processing import (
    aggregate_speed_matrix,
    nlanes_to_class,
    oneway_to_class,
    highway_to_class,
    build_line_graph_edge_index,
    build_split_data,
)
from src.models.multi_attr_gat import MultiAttrGAT
from src.models.saveing import load_checkpoint, save_checkpoint
from src.models.losses import compute_losses, compute_metrics, corrupt_inputs_with_flags
from src.models.masking import make_fixed_masks, bernoulli_mask
from src.models.scarcity import (
    compute_attr_availability, compute_neighbor_availability,
    evaluate_scarcity_bins, plot_scarcity_bins
)

# ─── Column offset constants (must match training script) ──────────────────────
CONT_LENGTH_COL    = 0
CONT_WIDTH_COL     = 1
CONT_MAX_COL       = 2
CONT_MIN_COL       = 3
CONT_AVG_START     = 4    # avg_speed_z  cols 4-15
CONT_AVG_END       = 16

CONT_LENMISS_COL   = 16
CONT_WIDMISS_COL   = 17
CONT_MAXMISS_COL   = 18
CONT_MINMISS_COL   = 19
CONT_AVGMISS_START = 20   # avg_speed_missing cols 20-31
CONT_AVGMISS_END   = 32

CONT_LENMASK_COL   = 32   # mask flags – set to 0 for inference (nothing is masked)
CONT_WIDMASK_COL   = 33
CONT_MAXMASK_COL   = 34
CONT_MINMASK_COL   = 35
CONT_AVGMASK_START = 36
CONT_AVGMASK_END   = 48


# ─── Helpers ───────────────────────────────────────────────────────────────────
def build_x_cont(edges, speed_matrix, len_scaler, wid_scaler, max_scaler, min_scaler, avg_scaler):
    """
    Reproduce the exact feature matrix used during training.
    NaN values in raw data are replaced with 0 and flagged in the *_missing columns.
    Mask-flag columns are all zero (nothing is being masked at inference time).
    """
    N = len(edges)
    avg_speed_flat = speed_matrix.reshape(N, -1).astype(np.float32)   # (N, 12)
 
    length_raw = edges["length"].to_numpy(dtype=np.float32)
    length_log = np.log1p(length_raw)
    width_raw  = edges["width"].to_numpy(dtype=np.float32)
    max_raw    = edges["max_speed"].to_numpy(dtype=np.float32)
    min_raw    = edges["min_speed"].to_numpy(dtype=np.float32)
 
    # Use the scalers that were fitted on training data
    avg_speed_z = avg_scaler.transform(avg_speed_flat).astype(np.float32)
    length_z    = len_scaler.transform(length_log).astype(np.float32)
    width_z     = wid_scaler.transform(width_raw).astype(np.float32)
    max_z       = max_scaler.transform(max_raw).astype(np.float32)
    min_z       = min_scaler.transform(min_raw).astype(np.float32)
 
    # Missing indicator flags (1 = was NaN in raw data)
    avg_speed_missing = np.isnan(avg_speed_flat).astype(np.float32)
    width_missing     = np.isnan(width_raw).astype(np.float32)
    length_missing    = np.zeros(N, dtype=np.float32)
    max_missing       = np.isnan(max_raw).astype(np.float32)
    min_missing       = np.isnan(min_raw).astype(np.float32)
 
    # Replace NaN with 0 after recording flags
    avg_speed_z = np.nan_to_num(avg_speed_z, nan=0.0)
    width_z     = np.nan_to_num(width_z,     nan=0.0)
    max_z       = np.nan_to_num(max_z,        nan=0.0)
    min_z       = np.nan_to_num(min_z,        nan=0.0)
 
    # For truly missing values, set mask flags = 1 so the model sees the same
    # input pattern it was trained on (artificially masked observed roads had
    # value=0, missing_flag=0, mask_flag=1). Leaving mask flags at 0 for
    # genuinely missing data puts them in an out-of-distribution pattern that
    # the model never received gradient signal for, producing garbage outputs.
    x_cont = np.column_stack([
        length_z,                                      # 0
        width_z,                                       # 1
        max_z,                                         # 2
        min_z,                                         # 3
        avg_speed_z,                                   # 4-15
        length_missing,                                # 16
        width_missing,                                 # 17
        max_missing,                                   # 18
        min_missing,                                   # 19
        avg_speed_missing,                             # 20-31
        np.zeros(N, dtype=np.float32),                 # 32  len_mask
        width_missing,                                 # 33  wid_mask  = missing flag
        max_missing,                                   # 34  max_mask  = missing flag
        min_missing,                                   # 35  min_mask  = missing flag
        avg_speed_missing,                             # 36-47 avg_mask = missing flags
    ]).astype(np.float32)
 
    return x_cont, avg_speed_flat


# ─── Fine-tuning helpers ─────────────────────────────────────────────────────
def restrict_masks(masks, subset_bool):
    """AND every mask in `masks` with a per-node boolean subset (e.g. "is train
    node"), broadcasting over the extra time-of-day dimension for avg_speed."""
    out = {}
    for k, v in masks.items():
        out[k] = v & subset_bool.unsqueeze(1) if v.dim() > 1 else v & subset_bool
    return out


def load_or_build_split(edges, N, pyg_data_dir, target_city, train_frac, val_frac, device):
    """Reuse the train/val/test node split saved by 1.2_split_train_test.py for
    this city if present; otherwise fall back to the same spatial (lat) split
    computed inline so the script still works without that pre-processing step."""
    train_path = os.path.join(pyg_data_dir, f"{target_city}_train_idx.npy")
    val_path   = os.path.join(pyg_data_dir, f"{target_city}_val_idx.npy")
    test_path  = os.path.join(pyg_data_dir, f"{target_city}_test_idx.npy")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_idx = np.load(train_path)
        val_idx   = np.load(val_path)
        test_idx  = np.load(test_path)
        if len(train_idx) + len(val_idx) + len(test_idx) == N:
            print(f"Loaded target-city split from {pyg_data_dir}  "
                  f"(train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})")
        else:
            print("Warning: saved split size doesn't match current edge count — rebuilding split inline.")
            train_idx = val_idx = test_idx = None
    else:
        train_idx = val_idx = test_idx = None

    if train_idx is None:
        n_train = int(round(train_frac * N))
        n_val   = int(round(val_frac * N))
        centroids = edges.geometry.centroid
        coord = centroids.y.to_numpy()  # 'lat' axis, matching 1.2_split_train_test.py default
        order = np.argsort(coord, kind="stable")
        train_idx = np.sort(order[:n_train]).astype(np.int64)
        val_idx   = np.sort(order[n_train:n_train + n_val]).astype(np.int64)
        test_idx  = np.sort(order[n_train + n_val:]).astype(np.int64)
        print(f"Built spatial train/val/test split inline  "
              f"(train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})")

    def to_bool(idx):
        b = torch.zeros(N, dtype=torch.bool, device=device)
        b[torch.from_numpy(idx).long().to(device)] = True
        return b

    return to_bool(train_idx), to_bool(val_idx), to_bool(test_idx)


def finetune_on_target(model, data, train_bool, val_bool, HIGHWAY_UNK_ID, device,
                        epochs=150, lr=1e-4, p_mask=0.30, patience=20, log_every=10):
    """Fine-tune a source-pretrained model on the target city's train split.

    Forward passes always run on the *full* target graph so the GAT still has
    every node (train + val + test) as message-passing context — only the
    loss/backward is restricted to train nodes, and only fixed, held-out masks
    on val nodes are used to decide when to stop. Test nodes are never touched
    here; they stay reserved for the final, honest evaluation in infer().
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    valid = {
        "hwy": data.y_highway != HIGHWAY_UNK_ID,
        "lan": data.y_nlanes != -1,
        "onw": ~torch.isnan(data.y_oneway),
        "wid": ~torch.isnan(data.y_width),
        "max": ~torch.isnan(data.y_max),
        "min": ~torch.isnan(data.y_min),
        "avg": ~torch.isnan(data.y_avg_speed),
    }

    # Fixed (non-resampled) validation masks so val loss is comparable epoch to epoch.
    val_masks = restrict_masks(
        make_fixed_masks(data, p_mask=p_mask, seed=999, hwy_unk_id=HIGHWAY_UNK_ID), val_bool
    )

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    print(f"\n=== Fine-tuning on target train split (n={int(train_bool.sum())}) "
          f"| validating on val split (n={int(val_bool.sum())}) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_masks = restrict_masks(
            {k: bernoulli_mask(v, p_mask) for k, v in valid.items()}, train_bool
        )
        x_cont, hwy_in, lan_in, onw_in = corrupt_inputs_with_flags(data, train_masks, HIGHWAY_UNK_ID)
        pred = model(x_cont, hwy_in, lan_in, onw_in, data.edge_index)
        train_total, train_losses = compute_losses(pred, data, train_masks, model, device)
        train_total.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            x_cont_v, hwy_v, lan_v, onw_v = corrupt_inputs_with_flags(data, val_masks, HIGHWAY_UNK_ID)
            pred_v = model(x_cont_v, hwy_v, lan_v, onw_v, data.edge_index)
            val_total, _ = compute_losses(pred_v, data, val_masks, model, device)
        val_loss = val_total.item()

        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % log_every == 0 or improved:
            print(f"  [ft] epoch {epoch:04d}/{epochs}  train_loss={train_total.item():.4f}  "
                  f"val_loss={val_loss:.4f}{'  *best*' if improved else ''}")

        if no_improve >= patience:
            print(f"  [ft] early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  [ft] restored best-val checkpoint  (val_loss={best_val:.4f})\n")
    return model


def infer(edges_path: str,
          speed_matrix_path: str,
          checkpoint_dir: str,
          source_city: str,
          target_city: str,
          output_path: str,
          plots_dir: str = "./plots/cross-city/",
          results_dir: str = "./results",
          n_bins: int = 5,
          device_str: str = "auto",
          pyg_data_dir: str = "./data/pyg_data/",
          finetune: bool = True,
          ft_epochs: int = 150,
          ft_lr: float = 1e-4,
          ft_p_mask: float = 0.30,
          ft_patience: int = 20,
          ft_train_frac: float = 0.85,
          ft_val_frac: float = 0.05,
          save_finetuned: bool = False):

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

     # ── Load checkpoint ───────────────────────────────────────────────────────
    # load_checkpoint instantiates + loads the model internally and returns a tuple
    epoch = 5000
    
    if source_city == 'chicago': epoch = 1400
    elif source_city == 'jakarta': epoch = 1100
    elif source_city == 'NewYorkCity': epoch = 1900
    elif source_city == 'sanFrancisco': epoch = 4400
    elif source_city == 'singapore': epoch = 3100
    elif source_city == 'washingtonDC': epoch = 3300
    
    ckpt_path = os.path.join(checkpoint_dir, f"{source_city}/gat_multitask_e{epoch}.pt")
    
    (model,
     hwy2id, id2hwy,
     HIGHWAY_MASK_ID, LANES_MASK_ID, LANES_MISS_ID,
     ONEWAY_MASK_ID, ONEWAY_MISS_ID,
     len_scaler, wid_scaler, max_scaler, min_scaler, avg_scaler,
     meta) = load_checkpoint(MultiAttrGAT, device=device_str, ckpt_path=ckpt_path)
 
    num_highway = len(hwy2id)
    cont_dim    = model.cont_dim if hasattr(model, "cont_dim") else 48
    print(f"Checkpoint loaded  |  num_highway={num_highway}  cont_dim={cont_dim}")

    # ── Load data ─────────────────────────────────────────────────────────────
    edges = gpd.read_parquet(edges_path)
    speed_matrix = np.load(speed_matrix_path)

    if edges.crs == "EPSG:4326":
        print("CRS: EPSG:4326 ✓")
    else:
        print(f"Warning: CRS is {edges.crs}, expected EPSG:4326")

    # ── Preprocess (mirrors training exactly) ─────────────────────────────────
    speed_matrix = aggregate_speed_matrix(
        speed_matrix, ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
    )

    edges["nlanes_cls"] = nlanes_to_class(edges["nlanes"])
    edges["oneway"]     = oneway_to_class(edges["oneway"])
    edges.to_crs(epsg=3857, inplace=True)

    highways_ids, _, _, _, _, _, _ = highway_to_class(
        edges["road_type"], hwy2id=hwy2id          # pass the saved mapping so IDs match
    )
    edges["highway_id"] = highways_ids

    edges = edges.reset_index().rename(columns={"index": "idx"})
    N = len(edges)

    # ── Build feature matrix ───────────────────────────────────────────────────
    x_cont_all, avg_speed_flat = build_x_cont(
        edges, speed_matrix, len_scaler, wid_scaler, max_scaler, min_scaler,avg_scaler
    )

    # ── Categorical inputs ─────────────────────────────────────────────────────
    nlanes_in_all = edges["nlanes_cls"].to_numpy(dtype=np.int64)
    nlanes_in_all = np.where(nlanes_in_all == -1, LANES_MISS_ID, nlanes_in_all).astype(np.int64)

    oneway_in_all = edges["oneway"].to_numpy(dtype=np.float32)
    oneway_in_all = np.where(np.isnan(oneway_in_all), ONEWAY_MISS_ID, oneway_in_all).astype(np.int64)

    # Regression targets z-scored with the training scalers (NaN preserved),
    # matching how the model was trained — the eval losses/metrics below
    # operate in z-space and MAEs are rescaled to original units via mae_scale.
    y_highway_all = edges["highway_id"].to_numpy(dtype=np.int64)
    y_nlanes_all  = edges["nlanes_cls"].to_numpy(dtype=np.int64)
    y_oneway_all  = edges["oneway"].to_numpy(dtype=np.float32)
    y_width_all   = wid_scaler.transform(edges["width"].to_numpy(dtype=np.float32)).astype(np.float32)
    y_max_all     = max_scaler.transform(edges["max_speed"].to_numpy(dtype=np.float32)).astype(np.float32)
    y_min_all     = min_scaler.transform(edges["min_speed"].to_numpy(dtype=np.float32)).astype(np.float32)
    y_avg_all     = avg_scaler.transform(avg_speed_flat).astype(np.float32)

    HIGHWAY_UNK_ID = hwy2id["__UNK__"]
    mae_scale = {"wid": wid_scaler.sd, "max": max_scaler.sd, "min": min_scaler.sd, "avg": avg_scaler.sd}

    # ── Build full graph (all nodes = "test" split with all indices) ───────────
    edge_index_full_np = build_line_graph_edge_index(
        edges, u_col="source", v_col="target", eid_col="idx"
    ).cpu().numpy()

    all_idx = np.arange(N, dtype=np.int64)

    data = build_split_data(
        all_idx,
        N,
        edge_index_full_np,
        x_cont_all,
        y_highway_all,
        nlanes_in_all,
        oneway_in_all,
        y_nlanes_all,
        y_oneway_all,
        y_width_all,
        y_max_all,
        y_min_all,
        y_avg_all,
        device,
    )
    print(f"Graph: {data.num_nodes} nodes | {data.edge_index.shape[1]} edges")

    # ── Target-city train/val/test split (test stays untouched by fine-tuning) ─
    train_bool, val_bool, test_bool = load_or_build_split(
        edges, N, pyg_data_dir, target_city, ft_train_frac, ft_val_frac, device
    )

    # ── Fine-tune the source-pretrained model on the target train split ───────
    if finetune:
        model = finetune_on_target(
            model, data, train_bool, val_bool, HIGHWAY_UNK_ID, device,
            epochs=ft_epochs, lr=ft_lr, p_mask=ft_p_mask, patience=ft_patience,
        )
        if save_finetuned:
            save_checkpoint(
                model=model, num_highway=num_highway, hwy2id=hwy2id, id2hwy=id2hwy,
                HIGHWAY_MASK_ID=HIGHWAY_MASK_ID, LANES_MASK_ID=LANES_MASK_ID, LANES_MISS_ID=LANES_MISS_ID,
                ONEWAY_MASK_ID=ONEWAY_MASK_ID, ONEWAY_MISS_ID=ONEWAY_MISS_ID,
                len_scaler=len_scaler, wid_scaler=wid_scaler, max_scaler=max_scaler,
                min_scaler=min_scaler, avg_scaler=avg_scaler,
                SEED=0, P_MASK=ft_p_mask, city=f"{source_city}_finetuned_on_{target_city}",
                cont_dim=cont_dim, optimizer=None, epoch=ft_epochs, path_sufx="",
            )

    # ── Forward pass (no masking – all inputs passed as-is) ───────────────────
    with torch.no_grad():
        x_cont    = data.x_cont
        highway_in = data.highway_in         # already int ids, no masking
        nlanes_in  = data.nlanes_in
        oneway_in  = data.oneway_in

        pred = model(x_cont, highway_in, nlanes_in, oneway_in, data.edge_index)
    
    # ── Evaluate with masked inputs (true imputation quality) ────────────────
    # Randomly mask 30% of observed roads (matching training p_mask) before the
    # forward pass. This hides the value being predicted while leaving the other
    # 70% visible to the GAT as context — the same conditions the model trained
    # under. Masking 100% would be harder than training and give inflated errors;
    # leaving inputs unmasked lets the model trivially invert the Z-score.
    #
    # Restricted to the held-out test split: those nodes were never touched
    # during fine-tuning (backward pass), so this is an honest read of
    # generalization rather than train-set performance.
    with torch.no_grad():
        eval_masks = make_fixed_masks(data, p_mask=0.30, seed=2025, hwy_unk_id=HIGHWAY_UNK_ID)
        eval_masks = restrict_masks(eval_masks, test_bool)

        x_cont_eval, hwy_eval, lan_eval, onw_eval = corrupt_inputs_with_flags(
            data, eval_masks, HIGHWAY_UNK_ID
        )
        pred_eval = model(x_cont_eval, hwy_eval, lan_eval, onw_eval, data.edge_index)

        total_loss, losses = compute_losses(pred_eval, data, eval_masks, model, device)
        metrics            = compute_metrics(pred_eval, data, eval_masks, num_highway, mae_scale)

    print(f"\n=== Model evaluation on held-out TEST split only (n={int(test_bool.sum())}, "
          f"30% masked, matching training conditions) ===")
    print(f"  {'attribute':<14}  {'loss':>8}  {'metric':>30}  {'n_eval':>8}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*30}  {'-'*8}")
    n_obs = {k: int(eval_masks[k].sum()) for k in eval_masks}
    rows = [
        ("highway",   "hwy", f"macro-F1 = {metrics['hwy_macro_f1']:.4f}"),
        ("nlanes",    "lan", f"macro-F1 = {metrics['lan_macro_f1']:.4f}"),
        ("oneway",    "onw", f"AUROC    = {metrics['onw_auroc']:.4f}"),
        ("width",     "wid", f"MAE (m)  = {metrics['wid_mae_m']:.4f}"),
        ("max_speed", "max", f"MAE      = {metrics['max_mae']:.4f}"),
        ("min_speed", "min", f"MAE      = {metrics['min_mae']:.4f}"),
        ("avg_speed", "avg", f"MAE      = {metrics['avg_mae']:.4f}"),
    ]
    for label, key, metric_str in rows:
        print(f"  {label:<14}  {losses[key].item():>8.4f}  {metric_str:>30}  {n_obs[key]:>8}")
    print(f"  {'TOTAL':<14}  {total_loss.item():>8.4f}")

    masked_max_idx = torch.where(eval_masks["max"])[0]
    if len(masked_max_idx) >= 5:
        picks = masked_max_idx[torch.linspace(0, len(masked_max_idx) - 1, 5).long()]
        pred_max = max_scaler.inverse_transform(pred_eval["max_speed"].detach()[picks].cpu().numpy())
        true_max = max_scaler.inverse_transform(data.y_max[picks].cpu().numpy())
        print(f"\n  max_speed sample (km/h):")
        print(f"  {'road_idx':>10}  {'pred':>8}  {'true':>8}  {'|err|':>8}")
        for i, p, t in zip(picks.cpu().numpy(), pred_max, true_max):
            print(f"  {i:>10}  {p:>8.1f}  {t:>8.1f}  {abs(p - t):>8.1f}")

    # ── Save evaluation results ───────────────────────────────────────────────
    eval_results = {
        "total_loss": total_loss.item(),
        "losses": {k: losses[k].item() for k in losses},
        "metrics": metrics,
        "n_obs": n_obs,
        "source_city": source_city,
    }

    os.makedirs(results_dir, exist_ok=True)
    eval_output_path = os.path.join(results_dir, f"{source_city}_2_{target_city}_eval_results.json")
    import json
    with open(eval_output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved eval results → {eval_output_path}")

    # ── Data-scarcity evaluation ───────────────────────────────────────────────
    # Bin road segments by how attribute-complete their line-graph neighbours
    # are (ground truth, independent of the 30% eval mask above), then report
    # each attribute's accuracy within each bin. Bin 0 = neighbours have the
    # least observed data — the data-scarce regions.
    os.makedirs(plots_dir, exist_ok=True)
    attr_avail = compute_attr_availability(
        nlanes_valid=(y_nlanes_all != -1),
        oneway_valid=~np.isnan(y_oneway_all),
        width_valid=~np.isnan(y_width_all),
        max_valid=~np.isnan(y_max_all),
        min_valid=~np.isnan(y_min_all),
        avg_valid_frac=1.0 - np.isnan(y_avg_all).mean(axis=1),
    )
    neighbor_avail = compute_neighbor_availability(
        data.edge_index, torch.from_numpy(attr_avail).to(device), N
    )

    scarcity_metrics_fn = lambda p, d, m: compute_metrics(p, d, m, num_highway, mae_scale)
    scarcity_df = evaluate_scarcity_bins(pred_eval, data, eval_masks, neighbor_avail, n_bins, scarcity_metrics_fn)
    print(f"\nData-scarcity evaluation ({n_bins} bins requested; bin 0 = neighbours "
          f"have the least observed data):")
    print(scarcity_df.to_string(index=False))
    scarcity_csv = os.path.join(plots_dir, f"{source_city}_2_{target_city}_scarcity_bins.csv")
    scarcity_df.to_csv(scarcity_csv, index=False)
    print(f"Saved scarcity-bin table to: {scarcity_csv}")

    scarcity_metric_titles = {
        "hwy_macro_f1": "Highway Macro-F1",
        "lan_macro_f1": "Lanes Macro-F1",
        "onw_auroc": "Oneway AUROC",
        "wid_mae_m": "Width MAE (m)",
        "max_mae": "Max Speed MAE",
        "min_mae": "Min Speed MAE",
        "avg_mae": "Avg Speed MAE",
    }
    plot_scarcity_bins(
        scarcity_df, os.path.join(plots_dir, f"{source_city}_2_{target_city}_"), scarcity_metric_titles
    )

    # ── Decode predictions ────────────────────────────────────────────────────
    # highway → argmax of logits with the __UNK__/__MASK__ token logits
    # suppressed, so the model can never emit a non-class token. (Training also
    # no longer supervises UNK targets, this is just a safety net.)
    hwy_logits = pred["highway"].clone()
    for tok in ("__UNK__", "__MASK__"):
        tid = hwy2id.get(tok)
        if tid is not None:
            hwy_logits[:, tid] = float("-inf")
    hwy_pred_ids  = hwy_logits.argmax(dim=-1).cpu().numpy()              # (N,)
    hwy_pred_names = [id2hwy.get(int(i), "unknown") for i in hwy_pred_ids]
 
    # nlanes   → argmax of logits; classes 1/2/3 = lane count, class 0 = more than 3 lanes
    lan_pred_ids   = pred["nlanes"].argmax(dim=-1).cpu().numpy()          # (N,)
    lan_pred_count = np.where(lan_pred_ids == 0, 4, lan_pred_ids).astype(np.int64)  # decode class 0 → 4 (= 4+)
 
    # oneway   → sigmoid score → round to 0/1
    onw_pred_prob = torch.sigmoid(pred["oneway"].squeeze(-1)).cpu().numpy()  # (N,)
    onw_pred      = (onw_pred_prob >= 0.5).astype(np.float32)
 
    # width / max_speed / min_speed / avg_speed
    # Model predicts z-scored targets — decode back to original units with the
    # scalers saved in the checkpoint.
    wid_pred_m = wid_scaler.inverse_transform(pred["width"].squeeze(-1).cpu().numpy())  # metres
    max_pred   = apply_speed_constraints(
        max_scaler.inverse_transform(pred["max_speed"].squeeze(-1).cpu().numpy()), target_city, "max_speed"
    )
    min_pred   = apply_speed_constraints(
        min_scaler.inverse_transform(pred["min_speed"].squeeze(-1).cpu().numpy()), target_city, "min_speed"
    )
    avg_pred   = avg_scaler.inverse_transform(pred["avg_speed"].cpu().numpy())  # (N, 12) km/h
 

    # ── Build output: fill NaN slots, keep observed values unchanged ──────────
    out = edges.copy()

    def fill_where_missing(series, predictions, mask_condition):
        """Replace NaN / -1 entries with model predictions."""
        result = series.copy()
        result[mask_condition] = predictions[mask_condition]
        return result

    missing_hwy = out["road_type"].isna()
    missing_lan = out["nlanes_cls"] == -1
    missing_onw = out["oneway"].isna()
    missing_wid = out["width"].isna()
    missing_max = out["max_speed"].isna()
    missing_min = out["min_speed"].isna()

    # Predicted columns (always written – useful to compare with observed)
    # TODO: Need to replace hwy_pred_ids into hwy_pred_names since ids are just something we use internally
    hwy_pred_names = [-1 if x == '__UNK__' else x for x in hwy_pred_names]
    hwy_pred_names = [-2 if x == '__MASK__' else x for x in hwy_pred_names]

    out["pred_road_type"] = np.array(hwy_pred_names, dtype=np.float64).astype(np.int64)
    out["pred_nlanes_cls"] = lan_pred_ids.astype(np.int64)
    out["pred_nlanes"]     = lan_pred_count        # actual lane count (4 = 4 or more)
    out["pred_oneway"]     = onw_pred
    out["pred_width"]      = wid_pred_m
    out["pred_max_speed"]  = max_pred
    out["pred_min_speed"]  = min_pred

    # Imputed columns (observed kept, missing replaced by model)
    out["imputed_road_type"]  = out["road_type"].where(~missing_hwy, np.array(hwy_pred_names, dtype=np.float64).astype(np.int64))
    out["imputed_nlanes_cls"] = out["nlanes_cls"].where(~missing_lan, pd.array(lan_pred_ids))
    out["imputed_nlanes"]     = out["nlanes"].where(~missing_lan, pd.array(lan_pred_count.astype(np.float64)))
    out["imputed_oneway"]     = out["oneway"].where(~missing_onw, pd.array(onw_pred))
    out["imputed_width"]      = out["width"].where(~missing_wid, pd.array(wid_pred_m))
    out["imputed_max_speed"]  = out["max_speed"].where(~missing_max, pd.array(max_pred))
    out["imputed_min_speed"]  = out["min_speed"].where(~missing_min, pd.array(min_pred))

    # avg_speed: (N, 12) → 12 separate imputed columns
    period_labels = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
    day_types     = ["weekday", "weekend"]
    col_idx = 0
    avg_speed_obs = avg_speed_flat  # (N, 12)  NaN = originally missing
    for dt in day_types:
        for period in period_labels:
            col_name = f"avg_speed_{dt}_{period}"
            obs      = avg_speed_obs[:, col_idx]
            pred_col = avg_pred[:, col_idx]
            missing  = np.isnan(obs)
            imputed  = np.where(missing, pred_col, obs)
            out[col_name]              = obs
            out[f"pred_{col_name}"]    = pred_col      # ← add this
            out[f"imputed_{col_name}"] = imputed
            col_idx += 1

    # ── Sample of predictions for truly missing max_speed ────────────────────
    missing_max_idx = np.where(missing_max.to_numpy())[0]
    print(f"\n=== max_speed predictions for truly missing roads (n={len(missing_max_idx)}) ===")
    if len(missing_max_idx) >= 50:
        rng = np.random.default_rng(seed=42)
        picks = rng.choice(missing_max_idx, size=50, replace=False)
        picks = np.sort(picks)
        print(f"  {'road_idx':>10}  {'max_speed':>10}  {'road_type':>14}  {'nlanes':>7}")
        for i in picks:
            print(f"  {i:>10}  {max_pred[i]:>10.1f}  {str(out['road_type'].iloc[i]):>14}  {str(out['nlanes'].iloc[i]):>7}")
    else:
        print(f"  fewer than 10 missing roads — showing all")
        for i in missing_max_idx:
            print(f"  road {i}  max_speed={max_pred[i]:.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out.to_parquet(output_path)
    print(f"\nSaved imputed GeoDataFrame → {output_path}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n=== Imputation summary ===")
    for col, mask in [
        ("road_type",  missing_hwy),
        ("nlanes_cls", missing_lan),
        ("oneway",     missing_onw),
        ("width",      missing_wid),
        ("max_speed",  missing_max),
        ("min_speed",  missing_min),
    ]:
        n_filled = int(mask.sum())
        print(f"  {col:<14}  filled {n_filled:>6} / {N} ({100*n_filled/N:.1f}%)")

    avg_missing_total = int(np.isnan(avg_speed_flat).sum())
    avg_total_cells   = avg_speed_flat.size
    print(f"  avg_speed        filled {avg_missing_total:>6} / {avg_total_cells} cells "
          f"({100*avg_missing_total/avg_total_cells:.1f}%)")

    return out


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Impute missing road attributes with trained MultiAttrGAT")
    p.add_argument("--source_city",            default="jakarta")
    p.add_argument("--target_city",            default="jakarta")
    p.add_argument("--data_dir",        default="./data/raw_data")
    p.add_argument("--output_dir",        default="./data/imputed_data")
    p.add_argument("--checkpoint_dir",  default="./checkpoints")
    p.add_argument("--plots_dir",       default="./plots/cross-city/",
                   help="Directory to save the data-scarcity evaluation table/plots")
    p.add_argument("--results_dir",     default="./results",
                   help="Directory to save the eval_results.json (use a distinct dir from the "
                        "non-finetuned run so results don't overwrite each other)")
    p.add_argument("--n_bins",          type=int, default=5,
                   help="Number of data-scarcity bins for the scarcity evaluation")
    # p.add_argument("--output",          default='./data/imputed_data/jakarta.parquet',
                #    help="Output parquet path. Defaults to <data_dir>/<city>_imputed.parquet")
    p.add_argument("--device",          default="cuda",
                   help="'auto', 'cpu', 'cuda', 'cuda:0', …")

    # ── Fine-tuning on the target city ─────────────────────────────────────
    p.add_argument("--pyg_data_dir",    default="./data/pyg_data/",
                   help="Directory with {city}_train_idx.npy/_val_idx.npy/_test_idx.npy from 1.2_split_train_test.py; "
                        "falls back to an inline spatial split if not found")
    p.add_argument("--no_finetune",     action="store_true",
                   help="Skip fine-tuning and evaluate the source checkpoint zero-shot on the target city")
    p.add_argument("--ft_epochs",       type=int,   default=150)
    p.add_argument("--ft_lr",           type=float, default=1e-4,
                   help="Fine-tuning learning rate (lower than the 1e-3 used for training from scratch)")
    p.add_argument("--ft_p_mask",       type=float, default=0.30)
    p.add_argument("--ft_patience",     type=int,   default=20,
                   help="Early-stop fine-tuning after this many epochs with no val-loss improvement")
    p.add_argument("--ft_train_frac",   type=float, default=0.85,
                   help="Only used for the inline split fallback")
    p.add_argument("--ft_val_frac",     type=float, default=0.05,
                   help="Only used for the inline split fallback")
    p.add_argument("--save_finetuned",  action="store_true",
                   help="Save the fine-tuned model as its own checkpoint under --checkpoint_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    edges_path  = os.path.join(args.data_dir, f"{args.target_city}_edges.parquet")
    speed_path  = os.path.join(args.data_dir, f"{args.target_city}_speed_matrix.npy")
    output_path = os.path.join(args.output_dir, f"{args.target_city}_imputedBy_{args.source_city}.parquet")

    infer(
        edges_path       = edges_path,
        speed_matrix_path= speed_path,
        checkpoint_dir   = args.checkpoint_dir,
        source_city      = args.source_city,
        target_city      = args.target_city,
        output_path      = output_path,
        plots_dir        = args.plots_dir,
        results_dir      = args.results_dir,
        n_bins           = args.n_bins,
        device_str       = args.device,
        pyg_data_dir     = args.pyg_data_dir,
        finetune         = not args.no_finetune,
        ft_epochs        = args.ft_epochs,
        ft_lr            = args.ft_lr,
        ft_p_mask        = args.ft_p_mask,
        ft_patience      = args.ft_patience,
        ft_train_frac    = args.ft_train_frac,
        ft_val_frac      = args.ft_val_frac,
        save_finetuned   = args.save_finetuned,
    )