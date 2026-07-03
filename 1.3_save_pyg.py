"""
Save train / val / test PyG Data objects and test masks for GraphMAE.

Feature vector layout  (x, 19 columns):
  col 0      length      metres,  always present
  col 1      width       metres,  -1 if missing
  col 2      max_speed   km/h,    -1 if missing
  col 3      min_speed   km/h,    -1 if missing
  cols 4-15  avg_speed   km/h,    -1 if missing  (12 time slots)
  col 16     highway_id  integer code from pd.Categorical, -1 if unknown
  col 17     nlanes      lane count as float,  -1 if missing
  col 18     oneway      0.0 / 1.0,            -1 if missing

Targets (y_*) preserve NaN so mask-based evaluation works correctly.
A highway_categories.json mapping  int -> type string  is saved alongside.
"""

import os
import sys
import argparse
import json

sys.path.append(os.path.expanduser("~/websites/mapedia"))

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch_geometric.data import Data

from src.processing import build_line_graph_edge_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fill(arr, val=-1.0):
    """Return float32 copy with NaN replaced by val."""
    arr = np.asarray(arr, dtype=np.float32)
    return np.where(np.isnan(arr), val, arr)


def build_split(split_idx, N, x_all, edge_index_np, targets):
    """Build a PyG Data object for one split (train / val / test)."""
    split_idx = np.asarray(split_idx, dtype=np.int64)

    # Remap global node IDs to split-local IDs
    map_arr = np.full(N, -1, dtype=np.int64)
    map_arr[split_idx] = np.arange(len(split_idx), dtype=np.int64)

    src, dst = edge_index_np[0], edge_index_np[1]
    keep = (map_arr[src] >= 0) & (map_arr[dst] >= 0)
    edge_index = torch.from_numpy(
        np.stack([map_arr[src[keep]], map_arr[dst[keep]]], axis=0)
    ).long()

    data = Data(
        x=torch.from_numpy(x_all[split_idx]).float(),
        edge_index=edge_index,
    )
    data.num_nodes = len(split_idx)

    for key, arr in targets.items():
        setattr(data, key, torch.from_numpy(arr[split_idx]))

    return data


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Save PyG Data for GraphMAE.")
    p.add_argument("--city",         type=str, required=True,
                   help="City name (must match file prefixes)")
    p.add_argument("--pyg_data_dir", type=str, default="./data/pyg_data/",
                   help="Directory containing the pre-processed npz / npy / parquet files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    city = args.city
    pdir = args.pyg_data_dir

    # ── Load raw arrays ──────────────────────────────────────────────────────
    # true_data: observed values before artificial masking (NaN = truly missing)
    with np.load(f'{pdir}/{city}_true_data.npz') as f:
        keys = list(f.files)
        # order: avg_speed, width, min_speed, max_speed, road_type, nlanes, oneway
        avg_speed, width, min_spd, max_spd, road_type, nlanes, oneway = [f[k] for k in keys]

    # masked_data: needed only for length (always present, never masked)
    with np.load(f'{pdir}/{city}_masked_data.npz') as f:
        keys = list(f.files)
        # order: avg_speed, width, min, max, road_type, nlanes, oneway, length
        length = f[keys[7]]

    # masks: True = value IS known for this node
    with np.load(f'{pdir}/{city}_masks.npz') as f:
        keys = list(f.files)
        # order: avg_speed, width, min, max, road_type, nlanes, oneway
        avg_mask, wid_mask, min_mask, max_mask, hwy_mask, lan_mask, onw_mask = [f[k] for k in keys]

    # Split indices
    train_idx = np.load(f'{pdir}/{city}_train_idx.npy')
    val_idx   = np.load(f'{pdir}/{city}_val_idx.npy')
    test_idx  = np.load(f'{pdir}/{city}_test_idx.npy')

    edges = gpd.read_parquet(f'{pdir}/{city}_edges.parquet')
    N = len(edges)
    print(f"Total nodes: {N}  |  train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    # ── Highway encoding ─────────────────────────────────────────────────────
    # pd.Categorical.codes automatically assigns -1 to NaN / unknown values
    road_type_series = pd.Series(
        road_type.flatten() if road_type.ndim > 1 else road_type
    ).astype(str).replace('nan', np.nan)

    highway_cat  = pd.Categorical(road_type_series)
    highway_ids  = highway_cat.codes.astype(np.int64)   # -1 for unknown
    categories   = highway_cat.categories.tolist()
    print(f"Highway types ({len(categories)}): {categories}")

    # ── Feature matrix ───────────────────────────────────────────────────────
    avg = np.asarray(avg_speed, dtype=np.float32)
    if avg.ndim == 1:
        avg = avg[:, None]   # [N, 1] if single slot

    x_all = np.column_stack([
        fill(length),                    # col 0:    length (m)
        fill(width),                     # col 1:    width (m)
        fill(max_spd),                   # col 2:    max_speed (km/h)
        fill(min_spd),                   # col 3:    min_speed (km/h)
        fill(avg),                       # cols 4-?: avg_speed slots (km/h)
        highway_ids.astype(np.float32),  # next col: highway_id (-1 = unknown)
        fill(nlanes),                    # nlanes (lane count)
        fill(oneway),                    # oneway (0.0 / 1.0)
    ]).astype(np.float32)

    print(f"Feature matrix: {x_all.shape}")  # expect [N, 19] for 12-slot avg_speed

    # ── Targets (NaN preserved — used for evaluation against mask) ────────────
    targets = {
        'y_highway':   highway_ids,                            # int64, -1 = unknown
        'y_width':     np.asarray(width,   dtype=np.float32), # NaN = missing
        'y_max':       np.asarray(max_spd, dtype=np.float32),
        'y_min':       np.asarray(min_spd, dtype=np.float32),
        'y_nlanes':    np.asarray(nlanes,  dtype=np.float32),
        'y_oneway':    np.asarray(oneway,  dtype=np.float32),
        'y_avg_speed': avg,                                    # [N, 12] or [N, 1]
    }

    # ── Line-graph edge index ────────────────────────────────────────────────
    edges = edges.reset_index().rename(columns={"index": "idx"})
    edge_index_np = build_line_graph_edge_index(
        edges, u_col="source", v_col="target", eid_col="idx"
    ).cpu().numpy()

    # ── Build split Data objects ──────────────────────────────────────────────
    data_train = build_split(train_idx, N, x_all, edge_index_np, targets)
    data_val   = build_split(val_idx,   N, x_all, edge_index_np, targets)
    data_test  = build_split(test_idx,  N, x_all, edge_index_np, targets)

    print(f"Train: {data_train.num_nodes} nodes  {data_train.edge_index.shape[1]} edges")
    print(f"Val:   {data_val.num_nodes} nodes  {data_val.edge_index.shape[1]} edges")
    print(f"Test:  {data_test.num_nodes} nodes  {data_test.edge_index.shape[1]} edges")

    # ── Test masks (already specific to test nodes, no indexing needed) ────────
    test_masks = {
        'avg': torch.from_numpy(avg_mask),
        'wid': torch.from_numpy(wid_mask),
        'hwy': torch.from_numpy(hwy_mask),
        'lan': torch.from_numpy(lan_mask),
        'min': torch.from_numpy(min_mask),
        'max': torch.from_numpy(max_mask),
        'onw': torch.from_numpy(onw_mask),
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    torch.save(data_train, f'{pdir}/{city}_train_pygData.pt')
    torch.save(data_val,   f'{pdir}/{city}_val_pygData.pt')
    torch.save(data_test,  f'{pdir}/{city}_test_pygData.pt')
    torch.save(test_masks, f'{pdir}/{city}_test_masks.pt')

    with open(f'{pdir}/{city}_highway_categories.json', 'w') as f:
        json.dump({str(i): cat for i, cat in enumerate(categories)}, f, indent=2)

    print(f"\nSaved to {pdir}/")
    print(f"  {city}_train_pygData.pt")
    print(f"  {city}_val_pygData.pt")
    print(f"  {city}_test_pygData.pt")
    print(f"  {city}_test_masks.pt")
    print(f"  {city}_highway_categories.json")


if __name__ == "__main__":
    main()
