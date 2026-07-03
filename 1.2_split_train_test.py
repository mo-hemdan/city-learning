import os
import sys
import argparse

sys.path.append(os.path.expanduser("~/websites/mapedia"))

import numpy as np
import pandas as pd
import torch
import geopandas as gpd
from torch.utils.tensorboard import SummaryWriter

from src.processing import (
    aggregate_speed_matrix, nlanes_to_class_np, oneway_to_class_np,
    highway_to_class_np, ZScaler, build_line_graph_edge_index,
    build_split_data, degree_stats
)
from src.models.multi_attr_gat_old import MultiAttrGAT
from src.models.plotting import plot_avgspeed_nans_per_bin, plot_results
from src.models.losses import (
    get_optimizer, corrupt_inputs_with_flags, compute_losses,
    evaluate_losses_only, compute_metrics, evaluate_with_masks
)
from src.models.masking import make_fixed_masks, bernoulli_mask
from src.models.saveing import save_checkpoint


# =========================
# 1) Arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiAttrGAT on a city road graph.")

    parser.add_argument("--city",        type=str,   required=True,          help="City name (must match parquet/npy filenames)")
    parser.add_argument("--device",      type=str,   default=None,           help="Device to use: 'cpu', 'cuda', 'cuda:0', etc. Defaults to cuda if available.")
    parser.add_argument("--data_dir",    type=str,   default="./data/raw_data/", help="Directory containing parquet and npy files")
    parser.add_argument("--pyg_data_dir",    type=str,   default="./data/pyg_data/", help="Directory containing parquet and npy files")

    parser.add_argument("--plots_dir",   type=str,   default="./plots/single-city/",     help="Directory to save plots")

    parser.add_argument("--epochs",      type=int,   default=500,            help="Number of training epochs")
    parser.add_argument("--p_mask",      type=float, default=0.30,           help="Masking probability")
    parser.add_argument("--eval_every",  type=int,   default=1,              help="Evaluate metrics every N epochs")
    parser.add_argument("--seed",        type=int,   default=42,             help="Random seed")

    parser.add_argument("--train_frac",  type=float, default=0.85,           help="Fraction of data for training")
    parser.add_argument("--val_frac",    type=float, default=0.05,           help="Fraction of data for validation")
    parser.add_argument("--test_frac",   type=float, default=0.10,           help="Fraction of data for test")
    parser.add_argument("--split_axis",  type=str,   default="lat",          choices=["lat", "lon"], help="Spatial split axis (inductive only)")
    parser.add_argument("--setting",     type=str,   default="inductive",    choices=["inductive", "transductive"],
                        help="inductive: spatial/geographic split (contiguous regions); "
                             "transductive: random node split (test nodes scattered for better neighbour access)")

    parser.add_argument("--resume",      type=str,   default=None,           help="Path to a checkpoint .pt file to resume training from")

    return parser.parse_args()


# =========================
# 2) Main
# =========================
def main():
    args = parse_args()

    # Device setup — must happen before any torch operations
    if args.device is not None:
        device = torch.device(args.device)
        # Set CUDA_VISIBLE_DEVICES only when a specific cuda index is given (e.g. cuda:2)
        if args.device.startswith("cuda:"):
            # gpu_index = args.device.split(":")[1]
            # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
            device = torch.device(args.device)  # after restricting visibility, always "cuda"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"City   : {args.city}")
    print(f"Device : {device}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version   : {torch.version.cuda}")

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.plots_dir, exist_ok=True)

    # =========================
    # 3) Load data
    # =========================
    edges = gpd.read_parquet(os.path.join(args.data_dir, f"{args.city}_edges.parquet"))
    if edges.crs == "EPSG:4326":
        print("Already in EPSG:4326")
    else:
        print(f"CRS is {edges.crs}, expected EPSG:4326")

    speed_matrix = np.load(os.path.join(args.data_dir, f"{args.city}_speed_matrix.npy"))

    def print_edge_states(edges):
        null_counts  = edges.isnull().sum()
        availability = (1 - edges.isnull().mean()) * 100
        print("After merge summary:")
        print(pd.DataFrame({"null_count": null_counts, "availability": availability}))
        print("Column types:")
        print(edges.info())
    print_edge_states(edges)

    # =========================
    # 4) Preprocessing
    # =========================
    speed_matrix = aggregate_speed_matrix(
        speed_matrix, ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
    )
    edges['nlanes'] = edges['nlanes'].replace(0.0, 1.0)
    edges.to_crs(epsg=3857, inplace=True)
    
    # =========================
    # 5) Node split
    # =========================
    N       = len(edges)
    n_train = int(round(args.train_frac * N))
    n_val   = int(round(args.val_frac   * N))

    if args.setting == "inductive":
        # Geographic/spatial split: sort by coordinate so each split is a
        # contiguous geographic region.  Tests true out-of-distribution generalisation.
        centroids = edges.geometry.centroid
        coord     = (centroids.x if args.split_axis == "lon" else centroids.y).to_numpy()
        order     = np.argsort(coord, kind="stable")
        train_idx = np.sort(order[:n_train]).astype(np.int64)
        val_idx   = np.sort(order[n_train : n_train + n_val]).astype(np.int64)
        test_idx  = np.sort(order[n_train + n_val :]).astype(np.int64)
        print(f"[Spatial split — inductive] axis={args.split_axis}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    else:
        # Random node split: test nodes are scattered across the full graph so
        # they have train-node neighbours and benefit from transductive message
        # passing.  A geographic split would cluster test nodes together, leaving
        # them with almost no train neighbours and making transductive meaningless.
        rng       = np.random.default_rng(args.seed)
        perm      = rng.permutation(N)
        train_idx = np.sort(perm[:n_train]).astype(np.int64)
        val_idx   = np.sort(perm[n_train : n_train + n_val]).astype(np.int64)
        test_idx  = np.sort(perm[n_train + n_val :]).astype(np.int64)
        print(f"[Random split — transductive] train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    assert len(set(train_idx) & set(val_idx))   == 0
    assert len(set(train_idx) & set(test_idx))  == 0
    assert len(set(val_idx)   & set(test_idx))  == 0
    
    # Saving the train, val and test indicies 
    np.save(os.path.join(args.pyg_data_dir, f"{args.city}_train_idx.npy"), train_idx)
    np.save(os.path.join(args.pyg_data_dir, f"{args.city}_val_idx.npy"), val_idx)
    np.save(os.path.join(args.pyg_data_dir, f"{args.city}_test_idx.npy"), test_idx)
    np.save(os.path.join(args.pyg_data_dir, f"{args.city}_speed_matrix.npy"), speed_matrix)
    edges.to_parquet(os.path.join(args.pyg_data_dir, f"{args.city}_edges.parquet"))
    
    # =========================
    # 5.a) Convert to Numpy
    # =========================
    
    length_raw = edges["length"].to_numpy(dtype=np.float32)
    width_true  = edges["width"].to_numpy(dtype=np.float32)
    max_true    = edges["max_speed"].to_numpy(dtype=np.float32)
    min_true    = edges["min_speed"].to_numpy(dtype=np.float32)
    road_type_true  = edges["road_type"].to_numpy(dtype=np.float32)
    nlanes_true    = edges["nlanes"].to_numpy(dtype=np.float32)
    oneway_true    = edges["oneway"].to_numpy(dtype=np.float32)
    avg_speed_flat_true = speed_matrix.reshape(N, -1).astype(np.float32)  # (N, 12)
    
    print('shapes:', length_raw.shape)
    print('width_true', width_true.shape)
    print('avg_speed_flat: ', avg_speed_flat_true.shape)
    
    # =========================
    # 5.b) Masking
    # =========================
    
    np.random.seed(999)
    
    def mask_idx_np(X, test_idx, p=0.3):
        nonnan_mask = ~np.isnan(X[test_idx])
        r = np.random.random(nonnan_mask.shape)
        test_mask = nonnan_mask & (r < p)

        X_masked = X.copy()
        if len(X_masked.shape) > 1: 
            rows, cols = np.where(test_mask)
            X_masked[test_idx[rows], cols] = np.nan
        else:   X_masked[test_idx[test_mask]] = np.nan
        
        return X_masked, test_mask
    
    avg_speed_flat, avg_speed_mask = mask_idx_np(avg_speed_flat_true, test_idx)
    width_raw, width_mask = mask_idx_np(width_true, test_idx)
    min_raw, min_mask = mask_idx_np(min_true, test_idx)
    max_raw, max_mask = mask_idx_np(max_true, test_idx)
    road_type_raw, road_type_mask = mask_idx_np(road_type_true, test_idx)
    nlanes_raw, nlanes_mask = mask_idx_np(nlanes_true, test_idx)
    oneway_raw, oneway_mask = mask_idx_np(oneway_true, test_idx)
    
    np.savez_compressed(f'./data/pyg_data/{args.city}_masked_data.npz', *[
        avg_speed_flat,
        width_raw,
        min_raw,
        max_raw,
        road_type_raw,
        nlanes_raw,
        oneway_raw,
        length_raw
    ])
    np.savez_compressed(f'./data/pyg_data/{args.city}_masks.npz', *[
        avg_speed_mask,
        width_mask,
        min_mask,
        max_mask,
        road_type_mask,
        nlanes_mask,
        oneway_mask
    ])
    np.savez_compressed(f'./data/pyg_data/{args.city}_true_data.npz', *[
        avg_speed_flat_true,
        width_true,
        min_true,
        max_true,
        road_type_true,
        nlanes_true,
        oneway_true
    ])
    
    print('shapes:', width_raw.shape)
    print('width_true', width_raw.shape)
    print('avg_speed_flat: ', avg_speed_flat.shape)
    print('masks: ', avg_speed_mask.shape)
    print('masks: ', nlanes_mask.shape)

if __name__ == "__main__":
    main()