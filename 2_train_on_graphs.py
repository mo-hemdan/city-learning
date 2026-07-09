import os
import sys
import argparse
from types import SimpleNamespace

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
from src.models.multi_attr_gat import MultiAttrGAT
from src.models.plotting import plot_avgspeed_nans_per_bin, plot_results
from src.models.losses import (
    get_optimizer, corrupt_inputs_with_flags, compute_losses,
    evaluate_losses_only, compute_metrics, evaluate_with_masks
)
from src.models.masking import make_fixed_masks, bernoulli_mask
from src.models.saveing import save_checkpoint


# =========================
# Transductive helpers
# =========================
def _slice_data(data, idx_t):
    """Return a SimpleNamespace with y_* and input attributes sliced to idx_t nodes."""
    return SimpleNamespace(
        num_nodes=len(idx_t),
        edge_index=data.edge_index,
        x_cont=data.x_cont[idx_t],
        highway_in=data.highway_in[idx_t],
        nlanes_in=data.nlanes_in[idx_t],
        oneway_in=data.oneway_in[idx_t],
        y_highway=data.y_highway[idx_t],
        y_nlanes=data.y_nlanes[idx_t],
        y_oneway=data.y_oneway[idx_t],
        y_width=data.y_width[idx_t],
        y_max=data.y_max[idx_t],
        y_min=data.y_min[idx_t],
        y_avg_speed=data.y_avg_speed[idx_t],
    )


def _corrupt_full_at_idx(data_full, idx_t, masks, highway_corrupt_id):
    """Clone the full graph inputs and corrupt only the idx_t nodes."""
    x_cont = data_full.x_cont.clone()
    highway_in = data_full.highway_in.clone()
    nlanes_in = data_full.nlanes_in.clone()
    oneway_in = data_full.oneway_in.clone()

    data_sub = SimpleNamespace(
        x_cont=x_cont[idx_t],
        highway_in=highway_in[idx_t],
        nlanes_in=nlanes_in[idx_t],
        oneway_in=oneway_in[idx_t],
    )
    xc_t, hw_t, nl_t, ow_t = corrupt_inputs_with_flags(data_sub, masks, highway_corrupt_id)

    x_cont[idx_t] = xc_t
    highway_in[idx_t] = hw_t
    nlanes_in[idx_t] = nl_t
    oneway_in[idx_t] = ow_t
    return x_cont, highway_in, nlanes_in, oneway_in


@torch.no_grad()
def _evaluate_transductive(model, data_full, idx_t, masks, num_highway, device, hwy_unk_id, mae_scale):
    """Forward on full graph; compute loss and metrics only on idx_t subset."""
    model.eval()
    x_cont, hw, nl, ow = _corrupt_full_at_idx(data_full, idx_t, masks, hwy_unk_id)
    pred_full = model(x_cont, hw, nl, ow, data_full.edge_index)
    pred = {k: v[idx_t] for k, v in pred_full.items()}
    data_sub = _slice_data(data_full, idx_t)
    total, losses = compute_losses(pred, data_sub, masks, model, device)
    metrics = compute_metrics(pred, data_sub, masks, num_highway, mae_scale)
    return total.item(), {k: v.item() for k, v in losses.items()}, metrics


@torch.no_grad()
def _evaluate_transductive_losses_only(model, data_full, idx_t, masks, device, hwy_unk_id):
    """Forward on full graph; compute loss only on idx_t subset."""
    model.eval()
    x_cont, hw, nl, ow = _corrupt_full_at_idx(data_full, idx_t, masks, hwy_unk_id)
    with torch.inference_mode():
        pred_full = model(x_cont, hw, nl, ow, data_full.edge_index)
    pred = {k: v[idx_t] for k, v in pred_full.items()}
    data_sub = _slice_data(data_full, idx_t)
    total, losses = compute_losses(pred, data_sub, masks, model, device)
    return total.item(), {k: v.item() for k, v in losses.items()}


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
    parser.add_argument("--split_axis",  type=str,   default="lat",          choices=["lat", "lon"], help="Spatial split axis")
    parser.add_argument("--ckpt_path_sufx",  type=str,   default="",   help="if you want your checkpoint different")
    parser.add_argument("--setting",    type=str,   default="inductive",    choices=["inductive", "transductive"],
                        help="inductive: separate subgraphs per split; transductive: single full graph, loss on train nodes only")

    parser.add_argument("--resume",      type=str,   default=None,           help="Path to a checkpoint .pt file to resume training from")
    
    parser.add_argument("--tb_logs_sufx", type=str, default="", help="suffex to the tensorboard directory")
    
    parser.add_argument("--n_conv_layers", type=int, default=2)
    
    parser.add_argument('--grids', action='store_true', help='Enable grids logging')

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

    with np.load(f'./data/pyg_data/{args.city}_masked_data.npz') as data:
        # Safely loads arrays in the exact order they were saved
        avg_speed_flat, width_raw, min_raw, max_raw, road_type_raw, nlanes_raw, oneway_raw, length_raw = [
            data[f] for f in data.files
        ]
    with np.load(f'./data/pyg_data/{args.city}_masks.npz') as data:
        # Safely loads arrays in the exact order they were saved
        avg_speed_mask, width_mask, min_mask, max_mask, road_type_mask, nlanes_mask, oneway_mask = [
            data[f] for f in data.files
        ]
    with np.load(f'./data/pyg_data/{args.city}_true_data.npz') as data:
        # Safely loads arrays in the exact order they were saved
        avg_speed_flat_true, width_true, min_true, max_true, road_type_true, nlanes_true, oneway_true = [
            data[f] for f in data.files
        ]
    
    train_idx = np.load(os.path.join(args.pyg_data_dir, f"{args.city}_train_idx.npy"))
    val_idx = np.load(os.path.join(args.pyg_data_dir, f"{args.city}_val_idx.npy"))
    test_idx = np.load(os.path.join(args.pyg_data_dir, f"{args.city}_test_idx.npy"))
                        
    edges = gpd.read_parquet(os.path.join(args.pyg_data_dir, f"{args.city}_edges.parquet"))
    N         = len(edges)
    
    # =========================
    # 5c) Attributes Preprocessing
    # =========================
    # Setting them to the true values and GAT will take care of the masks
    avg_speed_flat, width_raw, min_raw, max_raw, road_type_raw, nlanes_raw, oneway_raw = avg_speed_flat_true, width_true, min_true, max_true, road_type_true, nlanes_true, oneway_true
    
    nlanes_cls = nlanes_to_class_np(nlanes_raw)
    oneway_raw = oneway_to_class_np(oneway_raw)
    highways_ids, HIGHWAY_MASK_ID, hwy2id, id2hwy, unique_highways, MASK_TOKEN, UNK_TOKEN = \
        highway_to_class_np(road_type_raw)
    # edges["highway_id"] = highways_ids
    

    # =========================
    # 6) Feature engineering & normalization
    # =========================
    
    length_log = np.log1p(length_raw)

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
    width_missing     = np.isnan(width_raw).astype(np.float32)
    length_missing    = np.zeros(N, dtype=np.float32)
    max_missing       = np.isnan(max_raw).astype(np.float32)
    min_missing       = np.isnan(min_raw).astype(np.float32)

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
        # mask flags mirror missing flags so truly-missing inputs look the same
        # in training and at inference (see build_x_cont in 3_predict_on_graphs)
        np.zeros(N, dtype=np.float32),        # len_mask
        width_missing,                        # wid_mask
        max_missing,                          # max_mask
        min_missing,                          # min_mask
        avg_speed_missing,                    # avg_speed_mask
    ]).astype(np.float32)

    # =========================
    # 7) Build graph splits
    # =========================
    LANES_MASK_ID  = 4 # nlanes classes: 1,2,3 = lane count, 0 = more than 3 lanes + MASK=4 + MISSING=5
    LANES_MISS_ID  = 5
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

    # Regression targets are z-scored (NaN preserved). Training on raw km/h with
    # SmoothL1 (beta=1) puts almost every error in the linear regime → constant
    # gradients and median-collapse; z-scored targets keep the loss quadratic over
    # most of the range. Predictions are decoded back with the saved scalers.
    # NOTE: must recompute from the raw (NaN-preserving) arrays here rather than
    # reuse avg_speed_z/width_z/max_z/min_z — those were already nan_to_num'd to
    # 0.0 above for use as x_cont input features, and reusing them as targets
    # made every truly-missing label look like a valid "0" (decoding back to the
    # scaler's mean instead of staying NaN), corrupting the loss/metrics/masks.
    y_avg_speed_all = avg_scaler.transform(avg_speed_flat).astype(np.float32)
    y_highway_all   = highways_ids
    y_nlanes_all    = nlanes_cls
    y_oneway_all    = oneway_raw
    y_width_all     = wid_scaler.transform(width_raw).astype(np.float32)
    y_max_all       = max_scaler.transform(max_raw).astype(np.float32)
    y_min_all       = min_scaler.transform(min_raw).astype(np.float32)

    nlanes_in_all = np.where(nlanes_cls == -1, LANES_MISS_ID, nlanes_cls).astype(np.int64)

    oneway_in_all = np.where(np.isnan(oneway_raw), ONEWAY_MISS_ID, oneway_raw).astype(np.int64)

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
    test_masks = {
        'avg': torch.from_numpy(avg_speed_mask),
        'wid': torch.from_numpy(width_mask),
        'hwy': torch.from_numpy(road_type_mask),
        'lan': torch.from_numpy(nlanes_mask),
        'min': torch.from_numpy(min_mask),
        'max': torch.from_numpy(max_mask),
        'onw': torch.from_numpy(oneway_mask),    
    }
        #    data.avg_speed_mask = torch.from_numpy(test_masks['avg_speed']).float()
    #     data.width_mask = torch.from_numpy(test_masks['width']).float()
    #     data.road_type_mask = torch.from_numpy(test_masks['road_type']).float()
    #     data.nlanes_mask = torch.from_numpy(test_masks['nlanes']).float()
    #     data.min_mask = torch.from_numpy(test_masks['min']).float()
    #     data.max_mask = torch.from_numpy(test_masks['max']).float()
    #     data.oneway_mask = torch.from_numpy(test_masks['oneway']).float()
    true_vals = {
        'avg_speed': avg_speed_flat_true,
        'width': width_true,
        'road_type': road_type_true,
        'nlanes': nlanes_true,
        'min': min_true,
        'max': max_true,
        'oneway': oneway_true,  
    }

    data_train = build_split_data(train_idx, **split_kwargs)
    data_val   = build_split_data(val_idx,   **split_kwargs)
    data_test  = build_split_data(test_idx, **split_kwargs)

    # Save
    print('Saving PyG Data of Train, Val, and Test...')
    torch.save(data_train.cpu(), f'./data/pyg_data/{args.city}_train_pygData.pt')
    torch.save(data_val.cpu(), f'./data/pyg_data/{args.city}_val_pygData.pt')
    torch.save(data_test.cpu(), f'./data/pyg_data/{args.city}_test_pygData.pt')
    torch.save({k: v.cpu() for k, v in test_masks.items()}, f'./data/pyg_data/{args.city}_test_masks.pt')
    print('Files Saved!')
    data_train = data_train.to(device)
    data_val   = data_val.to(device)
    data_test  = data_test.to(device)
    test_masks = {k: v.to(device) for k, v in test_masks.items()}

    # =========================
    # 7b) Transductive setup
    # =========================
    print(f"Setting: {args.setting}")
    if args.setting == "transductive":
        data_full = build_split_data(np.arange(N, dtype=np.int64), **split_kwargs)
        data_full = data_full.to(device)
        train_idx_t = torch.from_numpy(train_idx).long().to(device)
        val_idx_t   = torch.from_numpy(val_idx).long().to(device)
        test_idx_t  = torch.from_numpy(test_idx).long().to(device)
        data_train_view = _slice_data(data_full, train_idx_t)
        data_val_view   = _slice_data(data_full, val_idx_t)
        data_test_view  = _slice_data(data_full, test_idx_t)
        print(f"Full graph: {data_full.num_nodes} nodes | {data_full.edge_index.shape[1]} edges")
        print(f"  Train subset: {len(train_idx_t)} | Val subset: {len(val_idx_t)} | Test subset: {len(test_idx_t)}")
        print("Degree stats (full):", degree_stats(data_full))
    else:
        print(f"Train graph: {data_train.num_nodes} nodes | {data_train.edge_index.shape[1]} edges")
        print(f"Val graph:   {data_val.num_nodes} nodes | {data_val.edge_index.shape[1]} edges")
        print(f"Test graph:  {data_test.num_nodes} nodes | {data_test.edge_index.shape[1]} edges")
        print("Degree stats:")
        print("  Train:", degree_stats(data_train))
        print("  Val:  ", degree_stats(data_val))
        print("  Test: ", degree_stats(data_test))

    _ref = data_train_view if args.setting == "transductive" else data_train
    print('Availability of avg_speed:')
    print('train non nans: ', np.sum(~np.isnan(_ref.y_avg_speed.cpu().numpy())))
    print('val non nans: ', np.sum(~np.isnan(data_val.y_avg_speed.cpu().numpy())))
    print('test non nans: ', np.sum(~np.isnan(data_test.y_avg_speed.cpu().numpy())))

    print('Availability of min_speed:')
    print('train non nans: ', np.sum(~np.isnan(_ref.y_min.cpu().numpy())))
    print('val non nans: ', np.sum(~np.isnan(data_val.y_min.cpu().numpy())))
    print('test non nans: ', np.sum(~np.isnan(data_test.y_min.cpu().numpy())))

    # =========================
    # 8) Model
    # =========================
    num_highway = len(hwy2id)
    HIGHWAY_UNK_ID = hwy2id[UNK_TOKEN]
    # MAE multipliers: metrics computed in z-space × sd = original units (m, km/h)
    mae_scale = {"wid": wid_scaler.sd, "max": max_scaler.sd, "min": min_scaler.sd, "avg": avg_scaler.sd}
    model = MultiAttrGAT(num_highway=num_highway, cont_dim=48, n_conv_layers=args.n_conv_layers).to(device)
    optimizer = get_optimizer(model.parameters())

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume}  (epoch {start_epoch - 1} → continuing from {start_epoch})")

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    tb_log_dir = os.path.join(_script_dir, f"tb_logs{args.ckpt_path_sufx}", args.city)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs → {tb_log_dir}")

    if args.setting == "transductive":
        val_masks_fixed = make_fixed_masks(data_val_view, p_mask=args.p_mask, seed=999, hwy_unk_id=HIGHWAY_UNK_ID)
    else:
        val_masks_fixed = make_fixed_masks(data_val, p_mask=args.p_mask, seed=999, hwy_unk_id=HIGHWAY_UNK_ID)

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

    # =========================
    # 9) Training loop
    # =========================
    _plot_ref = data_train if args.setting == "inductive" else data_train_view
    plot_avgspeed_nans_per_bin(_plot_ref, os.path.join(args.plots_dir, "avgspeed_nans_per_bin.png"))

    cont_dim = int(data_train.x_cont.shape[1]) if args.setting == "inductive" else int(data_full.x_cont.shape[1])
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # resolve the train-subset data object for this epoch
        _train_src = data_train_view if args.setting == "transductive" else data_train

        valid_hwy = (_train_src.y_highway != HIGHWAY_UNK_ID)  # never supervise UNK
        valid_lan = (_train_src.y_nlanes != -1)
        valid_onw = ~torch.isnan(_train_src.y_oneway)
        valid_wid = ~torch.isnan(_train_src.y_width)
        valid_max = ~torch.isnan(_train_src.y_max)
        valid_min = ~torch.isnan(_train_src.y_min)
        valid_avg = ~torch.isnan(_train_src.y_avg_speed)

        train_masks = {
            "hwy": bernoulli_mask(valid_hwy, args.p_mask),
            "lan": bernoulli_mask(valid_lan, args.p_mask),
            "onw": bernoulli_mask(valid_onw, args.p_mask),
            "wid": bernoulli_mask(valid_wid, args.p_mask),
            "max": bernoulli_mask(valid_max, args.p_mask),
            "min": bernoulli_mask(valid_min, args.p_mask),
            "avg": bernoulli_mask(valid_avg, args.p_mask),
        }

        if args.setting == "inductive":
            x_cont, highway_in, nlanes_in, oneway_in = corrupt_inputs_with_flags(
                data_train, train_masks, HIGHWAY_UNK_ID
            )
            pred = model(x_cont, highway_in, nlanes_in, oneway_in, data_train.edge_index)
            total_loss, losses = compute_losses(pred, data_train, train_masks, model, device)
        else:
            # Transductive: forward on full graph, corrupt only train nodes
            x_cont, highway_in, nlanes_in, oneway_in = _corrupt_full_at_idx(
                data_full, train_idx_t, train_masks, HIGHWAY_UNK_ID
            )
            pred_full = model(x_cont, highway_in, nlanes_in, oneway_in, data_full.edge_index)
            pred = {k: v[train_idx_t] for k, v in pred_full.items()}
            total_loss, losses = compute_losses(pred, data_train_view, train_masks, model, device)

        total_loss.backward()
        optimizer.step()

        if args.setting == "inductive":
            val_total, val_losses = evaluate_losses_only(
                model, data_val, val_masks_fixed, device, HIGHWAY_UNK_ID
            )
        else:
            val_total, val_losses = _evaluate_transductive_losses_only(
                model, data_full, val_idx_t, val_masks_fixed, device, HIGHWAY_UNK_ID
            )

        history["epoch"].append(epoch)
        history["train_total"].append(total_loss.item())
        history["val_total"].append(val_total)
        for k in ["hwy", "lan", "onw", "wid", "max", "min", "avg"]:
            history["train_losses"][k].append(losses[k].item())
            history["val_losses"][k].append(val_losses[k])
        history["log_vars"].append(model.log_vars.detach().cpu().numpy().copy())

        # ── TensorBoard: losses (every epoch) ────────────────────────────────
        writer.add_scalar("loss/train_total", total_loss.item(), epoch)
        writer.add_scalar("loss/val_total",   val_total,         epoch)
        for k in ["hwy", "lan", "onw", "wid", "max", "min", "avg"]:
            writer.add_scalar(f"loss_train/{k}", losses[k].item(), epoch)
            writer.add_scalar(f"loss_val/{k}",   val_losses[k],    epoch)
        log_vars_now = model.log_vars.detach().cpu().numpy()
        for i, (k, lv) in enumerate(zip(["hwy", "lan", "onw", "wid", "max", "min", "avg"], log_vars_now)):
            writer.add_scalar(f"log_vars/{k}", float(lv), epoch)

        do_metrics = (epoch == 1) or (epoch % args.eval_every == 0)
        if do_metrics:
            train_metrics = compute_metrics(pred, _train_src, train_masks, num_highway, mae_scale)
            if args.setting == "inductive":
                _, _, val_metrics = evaluate_with_masks(
                    model, data_val, val_masks_fixed, num_highway, device, HIGHWAY_UNK_ID, mae_scale
                )
            else:
                _, _, val_metrics = _evaluate_transductive(
                    model, data_full, val_idx_t, val_masks_fixed, num_highway, device, HIGHWAY_UNK_ID, mae_scale
                )

            history["metric_epoch"].append(epoch)
            for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]:
                history["train_metrics"][k].append(train_metrics[k])
                history["val_metrics"][k].append(val_metrics[k])

            # ── TensorBoard: metrics (every eval_every epochs) ────────────────
            for metric_key in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]:
                writer.add_scalar(f"train/{metric_key}", train_metrics[metric_key], epoch)
                writer.add_scalar(f"val/{metric_key}",   val_metrics[metric_key],   epoch)

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

            if epoch > 50:
                masked_max_idx = torch.where(train_masks["max"])[0]
                if len(masked_max_idx) >= 5:
                    picks = masked_max_idx[
                        torch.linspace(0, len(masked_max_idx) - 1, 5).long()
                    ]
                    pred_max = max_scaler.inverse_transform(pred["max_speed"].detach()[picks].cpu().numpy())
                    true_max = max_scaler.inverse_transform(_train_src.y_max[picks].cpu().numpy())
                    print(f"  max_speed sample (masked observed roads, km/h):")
                    print(f"  {'road_idx':>10}  {'max_speed':>10}  {'true':>8}  {'|err|':>8}")
                    for i, p, t in zip(picks.cpu().numpy(), pred_max, true_max):
                        print(f"  {i:>10}  {p:>10.1f}  {t:>8.1f}  {abs(p - t):>8.1f}")

                truly_missing_idx = torch.where(torch.isnan(_train_src.y_max))[0]
                if len(truly_missing_idx) >= 10:
                    picks_m = truly_missing_idx[
                        torch.randperm(len(truly_missing_idx), generator=torch.Generator().manual_seed(42))[:10]
                    ].sort().values
                    pred_missing = max_scaler.inverse_transform(pred["max_speed"].detach()[picks_m].cpu().numpy())
                    print(f"  max_speed sample (truly missing roads, km/h):")
                    print(f"  {'road_idx':>10}  {'max_speed':>10}")
                    for i, p in zip(picks_m.cpu().numpy(), pred_missing):
                        print(f"  {i:>10}  {p:>10.1f}")

            if epoch % 100 == 0:
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
                    optimizer=optimizer,
                    epoch=epoch,
                    path_sufx=args.ckpt_path_sufx
                )
    # =========================
    # 10) Final evaluation
    # =========================
    train_metrics = compute_metrics(pred, _train_src, train_masks, num_highway, mae_scale)

    if args.setting == "inductive":
        test_masks_fixed = make_fixed_masks(data_test, p_mask=args.p_mask, seed=999, hwy_unk_id=HIGHWAY_UNK_ID)
        _, _, test_metrics = evaluate_with_masks(model, data_test, test_masks_fixed, num_highway, device, HIGHWAY_UNK_ID, mae_scale)
        _, _, val_metrics  = evaluate_with_masks(model, data_val,  val_masks_fixed,  num_highway, device, HIGHWAY_UNK_ID, mae_scale)
    else:
        test_masks_fixed = make_fixed_masks(data_test_view, p_mask=args.p_mask, seed=999, hwy_unk_id=HIGHWAY_UNK_ID)
        _, _, test_metrics = _evaluate_transductive(model, data_full, test_idx_t, test_masks_fixed, num_highway, device, HIGHWAY_UNK_ID, mae_scale)
        _, _, val_metrics  = _evaluate_transductive(model, data_full, val_idx_t,  val_masks_fixed,  num_highway, device, HIGHWAY_UNK_ID, mae_scale)

    print(
        f"\nFinal Results — {args.city}\n"
        f"VAL  : hwy_F1={val_metrics['hwy_macro_f1']:.3f}, lan_F1={val_metrics['lan_macro_f1']:.3f}, "
        f"onw_AUROC={val_metrics['onw_auroc']:.3f}, wid_MAE={val_metrics['wid_mae_m']:.3f}, "
        f"max_MAE={val_metrics['max_mae']:.3f}, min_MAE={val_metrics['min_mae']:.3f}\n"
        f"TEST : hwy_F1={test_metrics['hwy_macro_f1']:.3f}, lan_F1={test_metrics['lan_macro_f1']:.3f}, "
        f"onw_AUROC={test_metrics['onw_auroc']:.3f}, wid_MAE={test_metrics['wid_mae_m']:.3f}, "
        f"max_MAE={test_metrics['max_mae']:.3f}, min_MAE={test_metrics['min_mae']:.3f}"
    )

    if args.setting == "inductive":
        test_masks_fixed = make_fixed_masks(data_test, p_mask=args.p_mask, seed=2025, hwy_unk_id=HIGHWAY_UNK_ID)
        test_total, test_losses, test_metrics = evaluate_with_masks(
            model, data_test, test_masks_fixed, num_highway, device, HIGHWAY_UNK_ID, mae_scale
        )
    else:
        test_masks_fixed = make_fixed_masks(data_test_view, p_mask=args.p_mask, seed=2025, hwy_unk_id=HIGHWAY_UNK_ID)
        test_total, test_losses, test_metrics = _evaluate_transductive(
            model, data_full, test_idx_t, test_masks_fixed, num_highway, device, HIGHWAY_UNK_ID, mae_scale
        )
    print("TEST fixed-mask metrics:", test_metrics)
    print("TEST fixed-mask losses:",  test_losses)
    print('Running the agreed upon masks')

    if args.setting == "inductive":
        test_total, test_losses, test_metrics = evaluate_with_masks(
            model, data_test, test_masks, num_highway, device, HIGHWAY_UNK_ID, mae_scale
        )
    else:
        # test_masks has shape (n_test,) — matches data_test_view node count
        test_total, test_losses, test_metrics = _evaluate_transductive(
            model, data_full, test_idx_t, test_masks, num_highway, device, HIGHWAY_UNK_ID, mae_scale
        )
    print("TEST global-mask metrics:", test_metrics)
    print("TEST global-mask losses:",  test_losses)

    print(f"GPU memory — allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
          f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # ── TensorBoard: final test metrics ──────────────────────────────────────
    for k in ["hwy_macro_f1", "lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]:
        writer.add_scalar(f"test/{k}", test_metrics[k], args.epochs)
    writer.close()

    plot_results(history, os.path.join(args.plots_dir, args.city))

    # =========================
    # 11) Save checkpoint
    # =========================
    # cont_dim = int(data_train.x_cont.shape[1]) if args.setting == "inductive" else int(data_full.x_cont.shape[1])
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
        optimizer=optimizer,
        epoch=args.epochs,
        path_sufx=args.ckpt_path_sufx
    )


if __name__ == "__main__":
    main()