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
import os
import sys
sys.path.append(os.path.expanduser("~/websites/mapedia"))

import numpy as np
import pandas as pd
import geopandas as gpd
import torch

from src.processing import (
    aggregate_speed_matrix,
    nlanes_to_class,
    oneway_to_class,
    highway_to_class,
    build_line_graph_edge_index,
    build_split_data,
)
from src.models.multi_attr_gat import MultiAttrGAT
from src.models.saveing import load_checkpoint
from src.models.losses import compute_losses, compute_metrics

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
    width_missing     = np.isnan(width_z).astype(np.float32)
    length_missing    = np.zeros(N, dtype=np.float32)
    max_missing       = np.isnan(max_z).astype(np.float32)
    min_missing       = np.isnan(min_z).astype(np.float32)
 
    # Replace NaN with 0 after recording flags
    avg_speed_z = np.nan_to_num(avg_speed_z, nan=0.0)
    width_z     = np.nan_to_num(width_z,     nan=0.0)
    max_z       = np.nan_to_num(max_z,        nan=0.0)
    min_z       = np.nan_to_num(min_z,        nan=0.0)
 
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
        np.zeros(N, dtype=np.float32),                 # 32  len_mask  (all 0 at inference)
        np.zeros(N, dtype=np.float32),                 # 33  wid_mask
        np.zeros(N, dtype=np.float32),                 # 34  max_mask
        np.zeros(N, dtype=np.float32),                 # 35  min_mask
        np.zeros((N, 12), dtype=np.float32),           # 36-47 avg_mask
    ]).astype(np.float32)
 
    return x_cont, avg_speed_flat

def infer(edges_path: str,
          speed_matrix_path: str,
          checkpoint_dir: str,
          source_city: str,
          target_city: str,
          output_path: str,
          device_str: str = "auto"):

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

     # ── Load checkpoint ───────────────────────────────────────────────────────
    # load_checkpoint instantiates + loads the model internally and returns a tuple
    ckpt_path = os.path.join(checkpoint_dir, f"{source_city}_gat_multitask.pt")
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

    y_highway_all = edges["highway_id"].to_numpy(dtype=np.int64)
    y_nlanes_all  = edges["nlanes_cls"].to_numpy(dtype=np.int64)
    y_oneway_all  = edges["oneway"].to_numpy(dtype=np.float32)
    y_width_all   = edges["width"].to_numpy(dtype=np.float32)
    y_max_all     = edges["max_speed"].to_numpy(dtype=np.float32)
    y_min_all     = edges["min_speed"].to_numpy(dtype=np.float32)

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
        avg_speed_flat,
        device,
    )
    print(f"Graph: {data.num_nodes} nodes | {data.edge_index.shape[1]} edges")

    # ── Forward pass (no masking – all inputs passed as-is) ───────────────────
    with torch.no_grad():
        x_cont    = data.x_cont
        highway_in = data.highway_in         # already int ids, no masking
        nlanes_in  = data.nlanes_in
        oneway_in  = data.oneway_in

        pred = model(x_cont, highway_in, nlanes_in, oneway_in, data.edge_index)
    
    # ── Evaluate on all observed (non-missing) entries ───────────────────────
    with torch.no_grad():
        eval_masks = {
            "hwy": torch.ones(N, dtype=torch.bool, device=device),
            "lan": (data.y_nlanes != -1),
            "onw": ~torch.isnan(data.y_oneway),
            "wid": ~torch.isnan(data.y_width),
            "max": ~torch.isnan(data.y_max),
            "min": ~torch.isnan(data.y_min),
            "avg": ~torch.isnan(data.y_avg_speed),
        }

        total_loss, losses = compute_losses(pred, data, eval_masks, model, device)
        metrics            = compute_metrics(pred, data, eval_masks, num_highway)

    print("\n=== Model evaluation on observed entries ===")
    print(f"  {'attribute':<14}  {'loss':>8}  {'metric':>30}  {'n_obs':>8}")
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
    
    # ── Save evaluation results ───────────────────────────────────────────────
    eval_results = {
        "total_loss": total_loss.item(),
        "losses": {k: losses[k].item() for k in losses},
        "metrics": metrics,
        "n_obs": n_obs,
        "source_city": source_city,
    }

    eval_output_path = os.path.join('results', f"{source_city}_2_{target_city}_eval_results.json")
    import json
    with open(eval_output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved eval results → {eval_output_path}")

    # ── Decode predictions ────────────────────────────────────────────────────
    # TODO: Huge problem, we are predicting __UNK__ tokens to highways, pred['highway'] contains __UNK__ tokens for certain roads
    
    # highway  → argmax of logits
    hwy_pred_ids  = pred["highway"].argmax(dim=-1).cpu().numpy()          # (N,)
    hwy_pred_names = [id2hwy.get(int(i), "unknown") for i in hwy_pred_ids]
 
    # nlanes   → argmax of logits (class index 0/1/2)
    lan_pred_ids  = pred["nlanes"].argmax(dim=-1).cpu().numpy()           # (N,)
 
    # oneway   → sigmoid score → round to 0/1
    onw_pred_prob = torch.sigmoid(pred["oneway"].squeeze(-1)).cpu().numpy()  # (N,)
    onw_pred      = (onw_pred_prob >= 0.5).astype(np.float32)
 
    # width    → inverse-transform z-score → metres
    wid_pred_z    = pred["width"].squeeze(-1).cpu().numpy()               # (N,)
    wid_pred_m    = wid_scaler.inverse_transform(wid_pred_z)
 
    # max_speed / min_speed
    max_pred_z    = pred["max_speed"].squeeze(-1).cpu().numpy()
    max_pred      = max_scaler.inverse_transform(max_pred_z)
 
    min_pred_z    = pred["min_speed"].squeeze(-1).cpu().numpy()
    min_pred      = min_scaler.inverse_transform(min_pred_z)
 
    # avg_speed  → (N, 12) z-scores → raw km/h  (no avg_scaler saved; leave as z-scores)
    avg_pred_z    = pred["avg_speed"].cpu().numpy()                        # (N, 12)
    avg_pred_raw  = avg_pred_z  # inverse transform not available without avg_scaler
 

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
    out["pred_oneway"]     = onw_pred
    out["pred_width"]      = wid_pred_m
    out["pred_max_speed"]  = max_pred
    out["pred_min_speed"]  = min_pred

    # Imputed columns (observed kept, missing replaced by model)
    out["imputed_road_type"]  = out["road_type"].where(~missing_hwy, np.array(hwy_pred_names, dtype=np.float64).astype(np.int64))
    out["imputed_nlanes_cls"] = out["nlanes_cls"].where(~missing_lan, pd.array(lan_pred_ids))
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
            pred_col = avg_pred_raw[:, col_idx]
            missing  = np.isnan(obs)
            imputed  = np.where(missing, pred_col, obs)
            out[col_name]             = obs
            out[f"imputed_{col_name}"] = imputed
            col_idx += 1

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
    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n=== Imputation summary ===")
    print(f"  {'column':<14}  {'missing':>8}  {'before':>8}  {'after':>8}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}")
    for col, mask in [
        ("road_type",  missing_hwy),
        ("nlanes_cls", missing_lan),
        ("oneway",     missing_onw),
        ("width",      missing_wid),
        ("max_speed",  missing_max),
        ("min_speed",  missing_min),
    ]:
        n_filled     = int(mask.sum())
        avail_before = 100 * (N - n_filled) / N
        avail_after  = 100.0
        print(f"  {col:<14}  {n_filled:>7}  {avail_before:>7.1f}%  {avail_after:>7.1f}%")

    avg_missing_total = int(np.isnan(avg_speed_flat).sum())
    avg_total_cells   = avg_speed_flat.size
    avail_before_avg  = 100 * (avg_total_cells - avg_missing_total) / avg_total_cells
    print(f"  {'avg_speed':<14}  {avg_missing_total:>7}  {avail_before_avg:>7.1f}%  {'100.0':>7}%")
    return out


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Impute missing road attributes with trained MultiAttrGAT")
    p.add_argument("--source_city",            default="jakarta")
    p.add_argument("--target_city",            default="jakarta")
    p.add_argument("--data_dir",        default="./data/raw_data")
    p.add_argument("--checkpoint_dir",  default="./checkpoints")
    p.add_argument("--output",          default='./data/imputed_data/jakarta.parquet',
                   help="Output parquet path. Defaults to <data_dir>/<city>_imputed.parquet")
    p.add_argument("--device",          default="cuda",
                   help="'auto', 'cpu', 'cuda', 'cuda:0', …")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    edges_path  = os.path.join(args.data_dir, f"{args.target_city}_edges.parquet")
    speed_path  = os.path.join(args.data_dir, f"{args.target_city}_speed_matrix.npy")
    output_path = args.output or os.path.join(args.data_dir, f"{args.target_city}_imputedBy_{args.source_city}.parquet")

    infer(
        edges_path       = edges_path,
        speed_matrix_path= speed_path,
        checkpoint_dir   = args.checkpoint_dir,
        source_city      = args.source_city,
        target_city      = args.target_city,
        output_path      = output_path,
        device_str       = args.device,
    )