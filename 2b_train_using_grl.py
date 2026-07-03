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
    nlanes_to_class_np, oneway_to_class_np, highway_to_class_np,
    ZScaler, build_line_graph_edge_index, degree_stats
)
from src.models.highway_grl import HighwayGRL
from src.models.losses import get_optimizer
from src.models.losses_grl import (
    compute_losses_grl, compute_metrics_grl, valid_masks_grl,
    evaluate_losses_only_grl, evaluate_grl
)
from src.models.saveing_grl import save_checkpoint_grl
from src.models.plotting_grl import plot_results_grl
from src.models.scarcity import (
    compute_attr_availability, compute_neighbor_availability,
    evaluate_scarcity_bins, plot_scarcity_bins
)


# =========================
# Data helpers
# =========================
def build_split_data_grl(split_idx_np, N, edge_index_full_np, highway_all,
                          y_nlanes_all, y_oneway_all, y_width_all, y_max_all,
                          y_min_all, y_avg_speed_all, device):
    """Like src.processing.build_split_data, but only keeps the fields
    HighwayGRL needs (highway_in + targets) — no continuous features, no
    lane/oneway inputs, since this model never sees them."""
    map_arr = np.full(N, -1, dtype=np.int64)
    map_arr[split_idx_np] = np.arange(len(split_idx_np), dtype=np.int64)

    src_old, dst_old = edge_index_full_np[0], edge_index_full_np[1]
    keep = (map_arr[src_old] >= 0) & (map_arr[dst_old] >= 0)
    edge_index = torch.from_numpy(
        np.stack([map_arr[src_old[keep]], map_arr[dst_old[keep]]], axis=0)
    ).long()

    return SimpleNamespace(
        num_nodes=len(split_idx_np),
        edge_index=edge_index.to(device),
        highway_in=torch.from_numpy(highway_all[split_idx_np]).long().to(device),
        y_nlanes=torch.from_numpy(y_nlanes_all[split_idx_np]).long().to(device),
        y_oneway=torch.from_numpy(y_oneway_all[split_idx_np]).float().to(device),
        y_width=torch.from_numpy(y_width_all[split_idx_np]).float().to(device),
        y_max=torch.from_numpy(y_max_all[split_idx_np]).float().to(device),
        y_min=torch.from_numpy(y_min_all[split_idx_np]).float().to(device),
        y_avg_speed=torch.from_numpy(y_avg_speed_all[split_idx_np]).float().to(device),
    )


def _slice_data(data, idx_t):
    """Return a SimpleNamespace with y_* / highway_in attributes sliced to idx_t nodes."""
    return SimpleNamespace(
        num_nodes=len(idx_t),
        edge_index=data.edge_index,
        highway_in=data.highway_in[idx_t],
        y_nlanes=data.y_nlanes[idx_t],
        y_oneway=data.y_oneway[idx_t],
        y_width=data.y_width[idx_t],
        y_max=data.y_max[idx_t],
        y_min=data.y_min[idx_t],
        y_avg_speed=data.y_avg_speed[idx_t],
    )


@torch.no_grad()
def _evaluate_transductive_losses_only_grl(model, data_full, idx_t, device):
    """Forward on the full graph; compute loss only on the idx_t subset."""
    model.eval()
    pred_full = model(data_full.highway_in, data_full.edge_index)
    pred = {k: v[idx_t] for k, v in pred_full.items()}
    data_sub = _slice_data(data_full, idx_t)
    masks = valid_masks_grl(data_sub)
    total, losses = compute_losses_grl(pred, data_sub, masks, model, device)
    return total.item(), {k: v.item() for k, v in losses.items()}


@torch.no_grad()
def _evaluate_transductive_grl(model, data_full, idx_t, device, mae_scale=None):
    """Forward on the full graph; compute loss and metrics only on the idx_t subset."""
    model.eval()
    pred_full = model(data_full.highway_in, data_full.edge_index)
    pred = {k: v[idx_t] for k, v in pred_full.items()}
    data_sub = _slice_data(data_full, idx_t)
    masks = valid_masks_grl(data_sub)
    total, losses = compute_losses_grl(pred, data_sub, masks, model, device)
    metrics = compute_metrics_grl(pred, data_sub, masks, mae_scale)
    return total.item(), {k: v.item() for k, v in losses.items()}, metrics, pred, masks


# =========================
# Arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HighwayGRL (a GAT that only sees each road segment's "
                    "highway/road type) and gauge attribute-prediction accuracy in "
                    "data-scarce regions — road segments whose line-graph neighbours "
                    "have few observed attributes."
    )

    parser.add_argument("--city",         type=str,   required=True,               help="City name (must match parquet/npy/npz filenames)")
    parser.add_argument("--device",       type=str,   default=None,                help="Device to use: 'cpu', 'cuda', 'cuda:0', etc. Defaults to cuda if available.")
    parser.add_argument("--pyg_data_dir", type=str,   default="./data/pyg_data/",  help="Directory containing parquet/npy/npz files")
    parser.add_argument("--plots_dir",    type=str,   default="./plots/highway-grl/", help="Directory to save plots")

    parser.add_argument("--epochs",       type=int,   default=500,                 help="Number of training epochs")
    parser.add_argument("--eval_every",   type=int,   default=1,                   help="Evaluate metrics every N epochs")
    parser.add_argument("--seed",         type=int,   default=42,                  help="Random seed")

    parser.add_argument("--setting",      type=str,   default="inductive",         choices=["inductive", "transductive"],
                        help="inductive: separate subgraphs per split; transductive: single full graph, loss on train nodes only")

    parser.add_argument("--n_bins",       type=int,   default=5,                   help="Number of data-scarcity bins for the final evaluation")

    parser.add_argument("--resume",       type=str,   default=None,                help="Path to a checkpoint .pt file to resume training from")

    return parser.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"City   : {args.city}")
    print(f"Device : {device}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.plots_dir, exist_ok=True)

    # =========================
    # Load data
    # =========================
    with np.load(os.path.join(args.pyg_data_dir, f"{args.city}_masks.npz")) as data:
        avg_speed_mask, width_mask, min_mask, max_mask, road_type_mask, nlanes_mask, oneway_mask = [
            data[f] for f in data.files
        ]
    with np.load(os.path.join(args.pyg_data_dir, f"{args.city}_true_data.npz")) as data:
        avg_speed_flat_true, width_true, min_true, max_true, road_type_true, nlanes_true, oneway_true = [
            data[f] for f in data.files
        ]

    train_idx = np.load(os.path.join(args.pyg_data_dir, f"{args.city}_train_idx.npy"))
    val_idx = np.load(os.path.join(args.pyg_data_dir, f"{args.city}_val_idx.npy"))
    test_idx = np.load(os.path.join(args.pyg_data_dir, f"{args.city}_test_idx.npy"))

    edges = gpd.read_parquet(os.path.join(args.pyg_data_dir, f"{args.city}_edges.parquet"))
    N = len(edges)

    # =========================
    # Attribute preprocessing — HighwayGRL is only ever fed highway_in, so
    # every other attribute is a target only, always trained/evaluated against
    # the true (uncorrupted) value.
    # =========================
    nlanes_cls = nlanes_to_class_np(nlanes_true)
    oneway_cls = oneway_to_class_np(oneway_true)
    highways_ids, HIGHWAY_MASK_ID, hwy2id, id2hwy, unique_highways, MASK_TOKEN, UNK_TOKEN = \
        highway_to_class_np(road_type_true)

    avg_scaler = ZScaler()
    wid_scaler = ZScaler()
    max_scaler = ZScaler()
    min_scaler = ZScaler()

    avg_scaler.fit(avg_speed_flat_true[train_idx])
    wid_scaler.fit(width_true[train_idx])
    max_scaler.fit(max_true[train_idx])
    min_scaler.fit(min_true[train_idx])

    y_avg_speed_all = avg_scaler.transform(avg_speed_flat_true).astype(np.float32)
    y_width_all = wid_scaler.transform(width_true).astype(np.float32)
    y_max_all = max_scaler.transform(max_true).astype(np.float32)
    y_min_all = min_scaler.transform(min_true).astype(np.float32)
    y_nlanes_all = nlanes_cls
    y_oneway_all = oneway_cls
    y_highway_all = highways_ids

    # =========================
    # Line graph
    # =========================
    edges = edges.reset_index().rename(columns={"index": "idx"})
    edge_index_full = build_line_graph_edge_index(edges, u_col="source", v_col="target", eid_col="idx")
    edge_index_full_np = edge_index_full.cpu().numpy()

    split_kwargs = dict(
        N=N,
        edge_index_full_np=edge_index_full_np,
        highway_all=y_highway_all,
        y_nlanes_all=y_nlanes_all,
        y_oneway_all=y_oneway_all,
        y_width_all=y_width_all,
        y_max_all=y_max_all,
        y_min_all=y_min_all,
        y_avg_speed_all=y_avg_speed_all,
        device=device,
    )

    # Same fixed 30%-held-out test entries the baseline MultiAttrGAT is scored
    # on (produced by 1.2_split_train_test.py) — shape (n_test,), ordered like
    # the sorted test_idx array.
    global_test_masks = {
        'avg': torch.from_numpy(avg_speed_mask).to(device),
        'wid': torch.from_numpy(width_mask).to(device),
        'lan': torch.from_numpy(nlanes_mask).to(device),
        'min': torch.from_numpy(min_mask).to(device),
        'max': torch.from_numpy(max_mask).to(device),
        'onw': torch.from_numpy(oneway_mask).to(device),
    }

    # =========================
    # Data-scarcity proxy: how attribute-complete is each node's own record,
    # averaged over its line-graph neighbours (road segments sharing a junction).
    # =========================
    attr_avail = compute_attr_availability(
        nlanes_valid=~np.isnan(nlanes_true),
        oneway_valid=~np.isnan(oneway_true),
        width_valid=~np.isnan(width_true),
        max_valid=~np.isnan(max_true),
        min_valid=~np.isnan(min_true),
        avg_valid_frac=1.0 - np.isnan(avg_speed_flat_true).mean(axis=1),
    )
    attr_avail_t = torch.from_numpy(attr_avail).to(device)
    test_idx_t = torch.from_numpy(test_idx).long().to(device)

    data_train = build_split_data_grl(train_idx, **split_kwargs)
    data_val = build_split_data_grl(val_idx, **split_kwargs)
    data_test = build_split_data_grl(test_idx, **split_kwargs)

    if args.setting == "transductive":
        data_full = build_split_data_grl(np.arange(N, dtype=np.int64), **split_kwargs)
        train_idx_t = torch.from_numpy(train_idx).long().to(device)
        val_idx_t = torch.from_numpy(val_idx).long().to(device)

        data_train_view = _slice_data(data_full, train_idx_t)
        data_val_view = _slice_data(data_full, val_idx_t)
        data_test_view = _slice_data(data_full, test_idx_t)

        # Test-region neighbours may include train/val nodes too (full graph).
        neighbor_avail_full = compute_neighbor_availability(data_full.edge_index, attr_avail_t, N)
        neighbor_avail_test = neighbor_avail_full[test_idx_t]

        print(f"Full graph: {data_full.num_nodes} nodes | {data_full.edge_index.shape[1]} edges")
        print(f"  Train subset: {len(train_idx_t)} | Val subset: {len(val_idx_t)} | Test subset: {len(test_idx_t)}")
        print("Degree stats (full):", degree_stats(data_full))
    else:
        # Test region is a self-contained subgraph — only test-node neighbours count.
        neighbor_avail_test = compute_neighbor_availability(
            data_test.edge_index, attr_avail_t[test_idx_t], len(test_idx)
        )
        print(f"Train graph: {data_train.num_nodes} nodes | {data_train.edge_index.shape[1]} edges")
        print(f"Val graph:   {data_val.num_nodes} nodes | {data_val.edge_index.shape[1]} edges")
        print(f"Test graph:  {data_test.num_nodes} nodes | {data_test.edge_index.shape[1]} edges")
        print("Degree stats:")
        print("  Train:", degree_stats(data_train))
        print("  Val:  ", degree_stats(data_val))
        print("  Test: ", degree_stats(data_test))

    # =========================
    # Model
    # =========================
    num_highway = len(hwy2id)
    mae_scale = {"wid": wid_scaler.sd, "max": max_scaler.sd, "min": min_scaler.sd, "avg": avg_scaler.sd}
    model = HighwayGRL(num_highway=num_highway).to(device)
    optimizer = get_optimizer(model.parameters())

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = (ckpt.get("epoch") or 0) + 1
        print(f"Resumed from {args.resume}  (epoch {start_epoch - 1} → continuing from {start_epoch})")

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    tb_log_dir = os.path.join(_script_dir, "tb_logs_grl", args.city)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs → {tb_log_dir}")

    history = {
        "epoch": [], "train_total": [], "val_total": [],
        "train_losses": {k: [] for k in ["lan", "onw", "wid", "max", "min", "avg"]},
        "val_losses": {k: [] for k in ["lan", "onw", "wid", "max", "min", "avg"]},
        "metric_epoch": [],
        "train_metrics": {k: [] for k in ["lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]},
        "val_metrics": {k: [] for k in ["lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]},
    }

    _train_src = data_train_view if args.setting == "transductive" else data_train
    _val_src = data_val_view if args.setting == "transductive" else data_val

    # No input corruption/masking needed: HighwayGRL never sees these
    # attributes as input, so every valid (non-missing) label can be used for
    # supervision every epoch — the train/val/test node split already keeps
    # evaluation honest.
    train_masks = valid_masks_grl(_train_src)

    # =========================
    # Training loop
    # =========================
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        if args.setting == "inductive":
            pred = model(data_train.highway_in, data_train.edge_index)
        else:
            pred_full = model(data_full.highway_in, data_full.edge_index)
            pred = {k: v[train_idx_t] for k, v in pred_full.items()}

        total_loss, losses = compute_losses_grl(pred, _train_src, train_masks, model, device)
        total_loss.backward()
        optimizer.step()

        if args.setting == "inductive":
            val_total, val_losses = evaluate_losses_only_grl(model, data_val, device)
        else:
            val_total, val_losses = _evaluate_transductive_losses_only_grl(model, data_full, val_idx_t, device)

        history["epoch"].append(epoch)
        history["train_total"].append(total_loss.item())
        history["val_total"].append(val_total)
        for k in ["lan", "onw", "wid", "max", "min", "avg"]:
            history["train_losses"][k].append(losses[k].item())
            history["val_losses"][k].append(val_losses[k])

        writer.add_scalar("loss/train_total", total_loss.item(), epoch)
        writer.add_scalar("loss/val_total", val_total, epoch)
        for k in ["lan", "onw", "wid", "max", "min", "avg"]:
            writer.add_scalar(f"loss_train/{k}", losses[k].item(), epoch)
            writer.add_scalar(f"loss_val/{k}", val_losses[k], epoch)

        do_metrics = (epoch == 1) or (epoch % args.eval_every == 0)
        if do_metrics:
            train_metrics = compute_metrics_grl(pred, _train_src, train_masks, mae_scale)
            if args.setting == "inductive":
                _, _, val_metrics, _, _ = evaluate_grl(model, data_val, device, mae_scale)
            else:
                _, _, val_metrics, _, _ = _evaluate_transductive_grl(model, data_full, val_idx_t, device, mae_scale)

            history["metric_epoch"].append(epoch)
            for k in ["lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]:
                history["train_metrics"][k].append(train_metrics[k])
                history["val_metrics"][k].append(val_metrics[k])
                writer.add_scalar(f"train/{k}", train_metrics[k], epoch)
                writer.add_scalar(f"val/{k}", val_metrics[k], epoch)

            print(
                f"\n{'='*60}\n"
                f"  Epoch {epoch:04d}/{args.epochs}\n"
                f"{'='*60}\n"
                f"  LOSS   total={total_loss.item():.4f}\n"
                f"         lan={losses['lan'].item():.3f}  onw={losses['onw'].item():.3f}\n"
                f"         wid={losses['wid'].item():.3f}  max={losses['max'].item():.3f}  min={losses['min'].item():.3f}  avg={losses['avg'].item():.3f}\n"
                f"  VAL    lan_F1={val_metrics['lan_macro_f1']:.3f}  onw_AUROC={val_metrics['onw_auroc']:.3f}\n"
                f"         wid_MAE={val_metrics['wid_mae_m']:.3f}  max_MAE={val_metrics['max_mae']:.3f}  min_MAE={val_metrics['min_mae']:.3f}  avg_MAE={val_metrics['avg_mae']:.3f}\n"
                f"{'='*60}"
            )
            if epoch%50 == 0:
                save_checkpoint_grl(
                    model=model,
                    num_highway=num_highway,
                    hwy2id=hwy2id,
                    id2hwy=id2hwy,
                    wid_scaler=wid_scaler,
                    max_scaler=max_scaler,
                    min_scaler=min_scaler,
                    avg_scaler=avg_scaler,
                    SEED=args.seed,
                    city=args.city,
                    optimizer=optimizer,
                    epoch=args.epochs,
                )

    # =========================
    # Final evaluation
    # =========================
    if args.setting == "inductive":
        _test_src = data_test
        test_total, test_losses, test_metrics, test_pred, _ = evaluate_grl(model, data_test, device, mae_scale)
    else:
        _test_src = data_test_view
        test_total, test_losses, test_metrics, test_pred, _ = _evaluate_transductive_grl(
            model, data_full, test_idx_t, device, mae_scale
        )
    print("\nTEST (all valid labels) metrics:", test_metrics)
    print("TEST (all valid labels) losses: ", test_losses)

    test_global_metrics = compute_metrics_grl(test_pred, _test_src, global_test_masks, mae_scale)
    print("TEST (global-mask, comparable to baseline MultiAttrGAT) metrics:", test_global_metrics)

    if torch.cuda.is_available():
        print(f"GPU memory — allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
              f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    for k in ["lan_macro_f1", "onw_auroc", "wid_mae_m", "max_mae", "min_mae", "avg_mae"]:
        writer.add_scalar(f"test/{k}", test_metrics[k], args.epochs)
    writer.close()

    plot_results_grl(history, os.path.join(args.plots_dir, args.city))

    # =========================
    # Data-scarcity evaluation: bin test nodes by how attribute-rich their
    # neighbours are, and report per-attribute accuracy in each bin.
    # =========================
    scarcity_metrics_fn = lambda p, d, m: compute_metrics_grl(p, d, m, mae_scale)
    scarcity_df = evaluate_scarcity_bins(
        test_pred, _test_src, global_test_masks, neighbor_avail_test, args.n_bins, scarcity_metrics_fn
    )
    print(f"\nData-scarcity evaluation ({args.n_bins} bins requested; bin 0 = neighbours "
          f"have the least observed data):")
    print(scarcity_df.to_string(index=False))
    scarcity_csv = os.path.join(args.plots_dir, f"{args.city}_scarcity_bins.csv")
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
    plot_scarcity_bins(scarcity_df, os.path.join(args.plots_dir, args.city + "_"), scarcity_metric_titles)

    # =========================
    # Save checkpoint
    # =========================
    save_checkpoint_grl(
        model=model,
        num_highway=num_highway,
        hwy2id=hwy2id,
        id2hwy=id2hwy,
        wid_scaler=wid_scaler,
        max_scaler=max_scaler,
        min_scaler=min_scaler,
        avg_scaler=avg_scaler,
        SEED=args.seed,
        city=args.city,
        optimizer=optimizer,
        epoch=args.epochs,
    )


if __name__ == "__main__":
    main()
