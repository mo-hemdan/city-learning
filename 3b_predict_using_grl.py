"""
3b_predict_using_grl.py – Predict road attributes using a trained HighwayGRL
checkpoint (highway/road-type + graph structure only; no other attributes
seen as input), and report data-scarcity binned evaluation.

Usage
-----
    python 3b_predict_using_grl.py \
        --source_city jakarta \
        --target_city jakarta \
        --data_dir ./data/raw_data \
        --checkpoint_dir ./checkpoints \
        --plots_dir ./plots/cross-city-grl/
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.expanduser("~/websites/mapedia"))

from types import SimpleNamespace

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
)
from src.models.highway_grl import HighwayGRL
from src.models.saveing_grl import load_checkpoint_grl
from src.models.losses_grl import compute_losses_grl, compute_metrics_grl, make_fixed_masks_grl
from src.models.scarcity import (
    compute_attr_availability, compute_neighbor_availability,
    evaluate_scarcity_bins, plot_scarcity_bins
)


def infer(edges_path: str,
          speed_matrix_path: str,
          checkpoint_dir: str,
          source_city: str,
          target_city: str,
          output_path: str,
          plots_dir: str = "./plots/cross-city-grl/",
          n_bins: int = 5,
          device_str: str = "auto"):

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(checkpoint_dir, f"{source_city}_gat_highway_grl.pt")
    (model, hwy2id, id2hwy,
     wid_scaler, max_scaler, min_scaler, avg_scaler,
     meta) = load_checkpoint_grl(HighwayGRL, device=device_str, ckpt_path=ckpt_path)

    num_highway = len(hwy2id)
    print(f"Checkpoint loaded  |  num_highway={num_highway}")

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

    avg_speed_flat = speed_matrix.reshape(N, -1).astype(np.float32)

    # HighwayGRL's only input — the road/highway type.
    y_highway_all = edges["highway_id"].to_numpy(dtype=np.int64)

    # Everything else is a target only, z-scored with the training scalers
    # (NaN preserved) exactly like 3_predict_on_graphs.py.
    y_nlanes_all = edges["nlanes_cls"].to_numpy(dtype=np.int64)
    y_oneway_all = edges["oneway"].to_numpy(dtype=np.float32)
    y_width_all  = wid_scaler.transform(edges["width"].to_numpy(dtype=np.float32)).astype(np.float32)
    y_max_all    = max_scaler.transform(edges["max_speed"].to_numpy(dtype=np.float32)).astype(np.float32)
    y_min_all    = min_scaler.transform(edges["min_speed"].to_numpy(dtype=np.float32)).astype(np.float32)
    y_avg_all    = avg_scaler.transform(avg_speed_flat).astype(np.float32)

    mae_scale = {"wid": wid_scaler.sd, "max": max_scaler.sd, "min": min_scaler.sd, "avg": avg_scaler.sd}

    # ── Build full graph (all nodes) ────────────────────────────────────────────
    edge_index_full_np = build_line_graph_edge_index(
        edges, u_col="source", v_col="target", eid_col="idx"
    ).cpu().numpy()

    data = SimpleNamespace(
        num_nodes=N,
        edge_index=torch.from_numpy(edge_index_full_np).long().to(device),
        highway_in=torch.from_numpy(y_highway_all).long().to(device),
        y_nlanes=torch.from_numpy(y_nlanes_all).long().to(device),
        y_oneway=torch.from_numpy(y_oneway_all).float().to(device),
        y_width=torch.from_numpy(y_width_all).float().to(device),
        y_max=torch.from_numpy(y_max_all).float().to(device),
        y_min=torch.from_numpy(y_min_all).float().to(device),
        y_avg_speed=torch.from_numpy(y_avg_all).float().to(device),
    )
    print(f"Graph: {data.num_nodes} nodes | {data.edge_index.shape[1]} edges")

    # ── Forward pass ────────────────────────────────────────────────────────────
    with torch.no_grad():
        pred = model(data.highway_in, data.edge_index)

    # ── Evaluate on a fixed random 30% of observed labels ─────────────────────
    # HighwayGRL never takes these attributes as input (only highway_in), so
    # there is nothing to "hide" from the model — this just picks a reproducible
    # held-out slice of the ground truth to score against, analogous to the
    # baseline MultiAttrGAT's 30%-masked evaluation in 3_predict_on_graphs.py.
    with torch.no_grad():
        eval_masks = make_fixed_masks_grl(data, p_mask=0.30, seed=2025)
        total_loss, losses = compute_losses_grl(pred, data, eval_masks, model, device)
        metrics = compute_metrics_grl(pred, data, eval_masks, mae_scale)

    print("\n=== Model evaluation (30% held-out labels) ===")
    print(f"  {'attribute':<14}  {'loss':>8}  {'metric':>30}  {'n_eval':>8}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*30}  {'-'*8}")
    n_obs = {k: int(eval_masks[k].sum()) for k in eval_masks}
    rows = [
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
        pred_max = max_scaler.inverse_transform(pred["max_speed"].detach()[picks].cpu().numpy())
        true_max = max_scaler.inverse_transform(data.y_max[picks].cpu().numpy())
        print(f"\n  max_speed sample (km/h):")
        print(f"  {'road_idx':>10}  {'pred':>8}  {'true':>8}  {'|err|':>8}")
        for i, p, t in zip(picks.cpu().numpy(), pred_max, true_max):
            print(f"  {i:>10}  {p:>8.1f}  {t:>8.1f}  {abs(p - t):>8.1f}")

    # ── Save evaluation results ───────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    eval_results = {
        "total_loss": total_loss.item(),
        "losses": {k: losses[k].item() for k in losses},
        "metrics": metrics,
        "n_obs": n_obs,
        "source_city": source_city,
    }
    eval_output_path = os.path.join('results', f"{source_city}_2_{target_city}_highway_grl_eval_results.json")
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

    scarcity_metrics_fn = lambda p, d, m: compute_metrics_grl(p, d, m, mae_scale)
    scarcity_df = evaluate_scarcity_bins(pred, data, eval_masks, neighbor_avail, n_bins, scarcity_metrics_fn)
    print(f"\nData-scarcity evaluation ({n_bins} bins requested; bin 0 = neighbours "
          f"have the least observed data):")
    print(scarcity_df.to_string(index=False))
    scarcity_csv = os.path.join(plots_dir, f"{source_city}_2_{target_city}_scarcity_bins.csv")
    scarcity_df.to_csv(scarcity_csv, index=False)
    print(f"Saved scarcity-bin table to: {scarcity_csv}")

    scarcity_metric_titles = {
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

    missing_lan = out["nlanes_cls"] == -1
    missing_onw = out["oneway"].isna()
    missing_wid = out["width"].isna()
    missing_max = out["max_speed"].isna()
    missing_min = out["min_speed"].isna()

    # Predicted columns (always written – useful to compare with observed)
    out["pred_nlanes_cls"] = lan_pred_ids.astype(np.int64)
    out["pred_nlanes"]     = lan_pred_count        # actual lane count (4 = 4 or more)
    out["pred_oneway"]     = onw_pred
    out["pred_width"]      = wid_pred_m
    out["pred_max_speed"]  = max_pred
    out["pred_min_speed"]  = min_pred

    # Imputed columns (observed kept, missing replaced by model)
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
            out[f"pred_{col_name}"]    = pred_col
            out[f"imputed_{col_name}"] = imputed
            col_idx += 1

    # ── Save ──────────────────────────────────────────────────────────────────
    out.to_parquet(output_path)
    print(f"\nSaved imputed GeoDataFrame → {output_path}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n=== Imputation summary ===")
    for col, mask in [
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
    p = argparse.ArgumentParser(description="Impute missing road attributes with trained HighwayGRL")
    p.add_argument("--source_city",     default="jakarta")
    p.add_argument("--target_city",     default="jakarta")
    p.add_argument("--data_dir",        default="./data/raw_data")
    p.add_argument("--output_dir",      default="./data/imputed_data")
    p.add_argument("--checkpoint_dir",  default="./checkpoints")
    p.add_argument("--plots_dir",       default="./plots/cross-city-grl/",
                   help="Directory to save the data-scarcity evaluation table/plots")
    p.add_argument("--n_bins",          type=int, default=5,
                   help="Number of data-scarcity bins for the scarcity evaluation")
    p.add_argument("--device",          default="cuda",
                   help="'auto', 'cpu', 'cuda', 'cuda:0', …")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    edges_path  = os.path.join(args.data_dir, f"{args.target_city}_edges.parquet")
    speed_path  = os.path.join(args.data_dir, f"{args.target_city}_speed_matrix.npy")
    output_path = os.path.join(args.output_dir, f"{args.target_city}_imputedBy_{args.source_city}_highway_grl.parquet")

    infer(
        edges_path        = edges_path,
        speed_matrix_path = speed_path,
        checkpoint_dir    = args.checkpoint_dir,
        source_city       = args.source_city,
        target_city       = args.target_city,
        output_path       = output_path,
        plots_dir         = args.plots_dir,
        n_bins            = args.n_bins,
        device_str        = args.device,
    )
