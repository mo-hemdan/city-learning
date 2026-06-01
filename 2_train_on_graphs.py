import os
import sys
import argparse

sys.path.append(os.path.expanduser("~/websites/mapedia"))

import numpy as np
import pandas as pd
import torch
import geopandas as gpd

from src.processing import (
    aggregate_speed_matrix, nlanes_to_class, oneway_to_class,
    highway_to_class, ZScaler, build_line_graph_edge_index,
    build_split_data, degree_stats
)
from src.models.multi_attr_gat import MultiAttrGAT
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
    parser.add_argument("--plots_dir",   type=str,   default="./plots/single-city/",     help="Directory to save plots")

    parser.add_argument("--epochs",      type=int,   default=500,            help="Number of training epochs")
    parser.add_argument("--p_mask",      type=float, default=0.30,           help="Masking probability")
    parser.add_argument("--eval_every",  type=int,   default=1,              help="Evaluate metrics every N epochs")
    parser.add_argument("--seed",        type=int,   default=42,             help="Random seed")

    parser.add_argument("--train_frac",  type=float, default=0.85,           help="Fraction of data for training")
    parser.add_argument("--val_frac",    type=float, default=0.05,           help="Fraction of data for validation")
    parser.add_argument("--test_frac",   type=float, default=0.10,           help="Fraction of data for test")
    parser.add_argument("--split_axis",  type=str,   default="lat",          choices=["lat", "lon"], help="Spatial split axis")

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

    null_counts  = edges.isnull().sum()
    availability = (1 - edges.isnull().mean()) * 100
    print("After merge summary:")
    print(pd.DataFrame({"null_count": null_counts, "availability": availability}))
    print("Column types:")
    print(edges.info())

    # =========================
    # 4) Preprocessing
    # =========================
    speed_matrix = aggregate_speed_matrix(
        speed_matrix, ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
    )

    edges["nlanes_cls"] = nlanes_to_class(edges["nlanes"])
    edges["oneway"]     = oneway_to_class(edges["oneway"])
    edges.to_crs(epsg=3857, inplace=True)

    highways_ids, HIGHWAY_MASK_ID, hwy2id, id2hwy, unique_highways, MASK_TOKEN, UNK_TOKEN = \
        highway_to_class(edges["road_type"])
    edges["highway_id"] = highways_ids

    # =========================
    # 5) Spatial split
    # =========================
    N         = len(edges)
    centroids = edges.geometry.centroid
    coord     = (centroids.x if args.split_axis == "lon" else centroids.y).to_numpy()
    order     = np.argsort(coord, kind="stable")

    n_train   = int(round(args.train_frac * N))
    n_val     = int(round(args.val_frac   * N))

    train_idx = np.sort(order[:n_train]).astype(np.int64)
    val_idx   = np.sort(order[n_train : n_train + n_val]).astype(np.int64)
    test_idx  = np.sort(order[n_train + n_val :]).astype(np.int64)

    print(f"[Spatial split] axis={args.split_axis}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    assert len(set(train_idx) & set(val_idx))   == 0
    assert len(set(train_idx) & set(test_idx))  == 0
    assert len(set(val_idx)   & set(test_idx))  == 0

    # =========================
    # 6) Feature engineering & normalization
    # =========================
    avg_speed_flat = speed_matrix.reshape(N, -1).astype(np.float32)  # (N, 12)

    length_raw = edges["length"].to_numpy(dtype=np.float32)
    length_log = np.log1p(length_raw)
    width_raw  = edges["width"].to_numpy(dtype=np.float32)
    max_raw    = edges["max_speed"].to_numpy(dtype=np.float32)
    min_raw    = edges["min_speed"].to_numpy(dtype=np.float32)

    avg_scaler = ZScaler()
    len_scaler = ZScaler()
    wid_scaler = ZScaler()
    max_scaler = ZScaler()
    min_scaler = ZScaler()

    avg_scaler.fit(avg_speed_flat[train_idx])
    len_scaler.fit(length_log[train_idx])
    wid_scaler.fit(width_raw[train_idx])
    max_scaler.fit(max_raw[train_idx])
    min_scaler.fit(min_raw[train_idx])

    avg_speed_z = avg_scaler.transform(avg_speed_flat).astype(np.float32)
    length_z    = len_scaler.transform(length_log).astype(np.float32)
    width_z     = wid_scaler.transform(width_raw).astype(np.float32)
    max_z       = max_scaler.transform(max_raw).astype(np.float32)
    min_z       = min_scaler.transform(min_raw).astype(np.float32)

    avg_speed_missing = np.isnan(avg_speed_flat).astype(np.float32)
    width_missing     = np.isnan(width_z).astype(np.float32)
    length_missing    = np.zeros_like(width_missing, dtype=np.float32)
    max_missing       = np.isnan(max_z).astype(np.float32)
    min_missing       = np.isnan(min_z).astype(np.float32)

    avg_speed_z = np.nan_to_num(avg_speed_z, nan=0.0)
    width_z     = np.nan_to_num(width_z,     nan=0.0)
    max_z       = np.nan_to_num(max_z,       nan=0.0)
    min_z       = np.nan_to_num(min_z,       nan=0.0)

    x_cont_all = np.column_stack([
        length_z,           # (N,)
        width_z,            # (N,)
        max_z,              # (N,)
        min_z,              # (N,)
        avg_speed_z,        # (N, 12)
        length_missing,     # (N,)
        width_missing,      # (N,)
        max_missing,        # (N,)
        min_missing,        # (N,)
        avg_speed_missing,  # (N, 12)
        np.zeros(N, dtype=np.float32),        # len_mask
        np.zeros(N, dtype=np.float32),        # wid_mask
        np.zeros(N, dtype=np.float32),        # max_mask
        np.zeros(N, dtype=np.float32),        # min_mask
        np.zeros((N, 12), dtype=np.float32),  # avg_speed_mask
    ]).astype(np.float32)

    # =========================
    # 7) Build graph splits
    # =========================
    LANES_MASK_ID  = 3 # nlanes classes: 0,1,2 + MASK=3 + MISSING=4
    LANES_MISS_ID  = 4
    ONEWAY_MASK_ID = 2 # 0,1 + MASK=2 + MISSING=3
    ONEWAY_MISS_ID = 3

    edges = edges.reset_index().rename(columns={"index": "idx"})

    edge_index_full = build_line_graph_edge_index(
        edges, u_col="source", v_col="target", eid_col="idx"
    )

    ei   = edge_index_full
    pairs = (
        ei[0].cpu().numpy().astype(np.int64) * (ei.max().item() + 1)
        + ei[1].cpu().numpy().astype(np.int64)
    )
    dup = len(pairs) - len(np.unique(pairs))
    print(f"Duplicate directed edges in line-graph edge_index: {dup}")

    edge_index_full_np = edge_index_full.cpu().numpy()

    y_avg_speed_all = avg_speed_flat
    y_highway_all   = edges["highway_id"].to_numpy(dtype=np.int64)
    y_nlanes_all    = edges["nlanes_cls"].to_numpy(dtype=np.int64)
    y_oneway_all    = edges["oneway"].to_numpy(dtype=np.float32)
    y_width_all     = edges["width"].to_numpy(dtype=np.float32)
    y_max_all       = edges["max_speed"].to_numpy(dtype=np.float32)
    y_min_all       = edges["min_speed"].to_numpy(dtype=np.float32)

    nlanes_in_all = edges["nlanes_cls"].to_numpy(dtype=np.int64)
    nlanes_in_all = np.where(nlanes_in_all == -1, LANES_MISS_ID, nlanes_in_all).astype(np.int64)

    oneway_in_all = edges["oneway"].to_numpy(dtype=np.float32)
    oneway_in_all = np.where(np.isnan(oneway_in_all), ONEWAY_MISS_ID, oneway_in_all).astype(np.int64)

    split_kwargs = dict(
        N=N,
        edge_index_full_np=edge_index_full_np,
        x_cont_all=x_cont_all,
        y_highway_all=y_highway_all,
        nlanes_in_all=nlanes_in_all,
        oneway_in_all=oneway_in_all,
        y_nlanes_all=y_nlanes_all,
        y_oneway_all=y_oneway_all,
        y_width_all=y_width_all,
        y_max_all=y_max_all,
        y_min_all=y_min_all,
        y_avg_speed_all=y_avg_speed_all,
        device=device,
    )

    data_train = build_split_data(train_idx, **split_kwargs)
    data_val   = build_split_data(val_idx,   **split_kwargs)
    data_test  = build_split_data(test_idx,  **split_kwargs)

    print(f"Train graph: {data_train.num_nodes} nodes | {data_train.edge_index.shape[1]} edges")
    print(f"Val graph:   {data_val.num_nodes} nodes | {data_val.edge_index.shape[1]} edges")
    print(f"Test graph:  {data_test.num_nodes} nodes | {data_test.edge_index.shape[1]} edges")
    print("Degree stats:")
    print("  Train:", degree_stats(data_train))
    print("  Val:  ", degree_stats(data_val))
    print("  Test: ", degree_stats(data_test))
    
    print('Availability of avg_speed:')
    print('train non nans: ', np.sum(~np.isnan(data_train.y_avg_speed.cpu().numpy())))
    print('val non nans: ', np.sum(~np.isnan(data_val.y_avg_speed.cpu().numpy())))
    print('test non nans: ', np.sum(~np.isnan(data_test.y_avg_speed.cpu().numpy())))

    print('Availability of min_speed:')
    print('train non nans: ', np.sum(~np.isnan(data_train.y_min.cpu().numpy())))
    print('val non nans: ', np.sum(~np.isnan(data_val.y_min.cpu().numpy())))
    print('test non nans: ', np.sum(~np.isnan(data_test.y_min.cpu().numpy())))

    # =========================
    # 8) Model
    # =========================
    num_highway = len(hwy2id)
    model = MultiAttrGAT(num_highway=num_highway, cont_dim=48).to(device)
    optimizer = get_optimizer(model.parameters())

    # TODO: Remove the p_mask here, keep all validation as possible 
    val_masks_fixed = make_fixed_masks(data_val, p_mask=args.p_mask, seed=999)

    history = {
        "epoch":        [],
        "train_total":  [],
        "val_total":    [],
        "train_losses": {k: [] for k in ["hwy", "lan", "onw", "wid", "max", "min", "avg"]},
        "val_losses":   {k: [] for k in ["hwy", "lan", "onw", "wid", "max", "min", "avg"]},
        "metric_epoch": [],
        "train_metrics": {k: [] for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]},
        "val_metrics":   {k: [] for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]},
        "log_vars":     [],
    }

    # Column index constants
    CONT_LENGTH_COL   = 0
    CONT_WIDTH_COL    = 1
    CONT_MAX_COL      = 2
    CONT_MIN_COL      = 3
    CONT_AVG_START    = 4   # avg_speed_z occupies columns 4-15 (12 slots)
    CONT_AVG_END      = 16  # exclusive


    CONT_LENMISS_COL  = 4
    CONT_WIDMISS_COL  = 5
    CONT_MAXMISS_COL  = 6
    CONT_MINMISS_COL  = 7
    CONT_AVGMISS_START = 20  # avg_speed_missing: columns 20-31
    CONT_AVGMISS_END   = 32

    CONT_LENMASK_COL  = 8
    CONT_WIDMASK_COL  = 9
    CONT_MAXMASK_COL  = 10
    CONT_MINMASK_COL  = 11
    CONT_AVGMASK_START = 36  # avg_speed_mask: columns 36-47
    CONT_AVGMASK_END   = 48

    # =========================
    # 9) Training loop
    # =========================
    plot_avgspeed_nans_per_bin(data_train, os.path.join(args.plots_dir, "avgspeed_nans_per_bin.png"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        n = data_train.num_nodes

        valid_hwy = torch.ones(n, dtype=torch.bool, device=device)
        valid_lan = (data_train.y_nlanes != -1)
        valid_onw = ~torch.isnan(data_train.y_oneway)
        valid_wid = ~torch.isnan(data_train.y_width)
        valid_max = ~torch.isnan(data_train.y_max)
        valid_min = ~torch.isnan(data_train.y_min)
        valid_avg = ~torch.isnan(data_train.y_avg_speed)

        train_masks = {
            "hwy": bernoulli_mask(valid_hwy, args.p_mask),
            "lan": bernoulli_mask(valid_lan, args.p_mask),
            "onw": bernoulli_mask(valid_onw, args.p_mask),
            "wid": bernoulli_mask(valid_wid, args.p_mask),
            "max": bernoulli_mask(valid_max, args.p_mask),
            "min": bernoulli_mask(valid_min, args.p_mask),
            "avg": bernoulli_mask(valid_avg, args.p_mask),
        }

        x_cont, highway_in, nlanes_in, oneway_in = corrupt_inputs_with_flags(
            data_train, train_masks, HIGHWAY_MASK_ID
        )

        pred = model(x_cont, highway_in, nlanes_in, oneway_in, data_train.edge_index)

        total_loss, losses = compute_losses(pred, data_train, train_masks, model, device)
        total_loss.backward()
        optimizer.step()

        val_total, val_losses = evaluate_losses_only(
            model, data_val, val_masks_fixed, device, HIGHWAY_MASK_ID
        )

        history["epoch"].append(epoch)
        history["train_total"].append(total_loss.item())
        history["val_total"].append(val_total)
        for k in ["hwy", "lan", "onw", "wid", "max", "min", "avg"]:
            history["train_losses"][k].append(losses[k].item())
            history["val_losses"][k].append(val_losses[k])
        history["log_vars"].append(model.log_vars.detach().cpu().numpy().copy())

        do_metrics = (epoch == 1) or (epoch % args.eval_every == 0)
        if do_metrics:
            train_metrics = compute_metrics(pred, data_train, train_masks, num_highway)
            _, _, val_metrics = evaluate_with_masks(
                model, data_val, val_masks_fixed, num_highway, device, HIGHWAY_MASK_ID
            )

            history["metric_epoch"].append(epoch)
            for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]:
                history["train_metrics"][k].append(train_metrics[k])
                history["val_metrics"][k].append(val_metrics[k])

            log_vars = model.log_vars.detach().cpu().numpy()
            print(
                f"\n{'='*60}\n"
                f"  Epoch {epoch:04d}/{args.epochs}\n"
                f"{'='*60}\n"
                f"  LOSS   total={total_loss.item():.4f}\n"
                f"         hwy={losses['hwy'].item():.3f}  lan={losses['lan'].item():.3f}  onw={losses['onw'].item():.3f}\n"
                f"         wid={losses['wid'].item():.3f}  max={losses['max'].item():.3f}  min={losses['min'].item():.3f}  avg={losses['avg'].item():.3f}\n"
                f"  VAL    hwy_F1={val_metrics['hwy_macro_f1']:.3f}  lan_F1={val_metrics['lan_macro_f1']:.3f}  onw_AUROC={val_metrics['onw_auroc']:.3f}\n"
                f"         wid_MAE={val_metrics['wid_mae_m']:.3f}  max_MAE={val_metrics['max_mae']:.3f}  min_MAE={val_metrics['min_mae']:.3f}  avg_MAE={val_metrics['avg_mae']:.3f}\n"
                f"  VARS   {np.array2string(log_vars, precision=3, separator=', ')}\n"
                f"{'='*60}"
            )

    # =========================
    # 10) Final evaluation
    # =========================
    train_metrics     = compute_metrics(pred, data_train, train_masks, num_highway)
    test_masks_fixed  = make_fixed_masks(data_test, p_mask=args.p_mask, seed=999)
    _, _, test_metrics = evaluate_with_masks(model, data_test, test_masks_fixed, num_highway, device, HIGHWAY_MASK_ID)
    _, _, val_metrics  = evaluate_with_masks(model, data_val,  val_masks_fixed,  num_highway, device, HIGHWAY_MASK_ID)

    print(
        f"\nFinal Results — {args.city}\n"
        f"VAL  : hwy_F1={val_metrics['hwy_macro_f1']:.3f}, lan_F1={val_metrics['lan_macro_f1']:.3f}, "
        f"onw_AUROC={val_metrics['onw_auroc']:.3f}, wid_MAE={val_metrics['wid_mae_m']:.3f}, "
        f"max_MAE={val_metrics['max_mae']:.3f}, min_MAE={val_metrics['min_mae']:.3f}\n"
        f"TEST : hwy_F1={test_metrics['hwy_macro_f1']:.3f}, lan_F1={test_metrics['lan_macro_f1']:.3f}, "
        f"onw_AUROC={test_metrics['onw_auroc']:.3f}, wid_MAE={test_metrics['wid_mae_m']:.3f}, "
        f"max_MAE={test_metrics['max_mae']:.3f}, min_MAE={test_metrics['min_mae']:.3f}"
    )
    
    test_masks_fixed = make_fixed_masks(data_test, p_mask=args.p_mask, seed=2025)
    test_total, test_losses, test_metrics = evaluate_with_masks(
        model, data_test, test_masks_fixed, num_highway, device, HIGHWAY_MASK_ID
    )
    print("TEST fixed-mask metrics:", test_metrics)
    print("TEST fixed-mask losses:",  test_losses)

    print(f"GPU memory — allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
          f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    plot_results(history, os.path.join(args.plots_dir, args.city))

    # =========================
    # 11) Save checkpoint
    # =========================
    cont_dim = int(data_train.x_cont.shape[1])
    save_checkpoint(
        model=model,
        num_highway=num_highway,
        hwy2id=hwy2id,
        id2hwy=id2hwy,
        HIGHWAY_MASK_ID=HIGHWAY_MASK_ID,
        LANES_MASK_ID=LANES_MASK_ID,
        LANES_MISS_ID=LANES_MISS_ID,
        ONEWAY_MASK_ID=ONEWAY_MASK_ID,
        ONEWAY_MISS_ID=ONEWAY_MISS_ID,
        len_scaler=len_scaler,
        wid_scaler=wid_scaler,
        max_scaler=max_scaler,
        min_scaler=min_scaler,
        avg_scaler=avg_scaler,
        SEED=args.seed,
        P_MASK=args.p_mask,
        city=args.city,
        cont_dim=cont_dim,
    )


if __name__ == "__main__":
    main()