"""
Data-scarcity evaluation helpers, shared by the baseline MultiAttrGAT and the
HighwayGRL ablation. The idea: bin road segments (line-graph nodes) by how
attribute-complete their *neighbours* are, then report each attribute's
accuracy within each bin. Bin 0 = the neighbours have the least observed
data — the "data-scarce" regions we care about.
"""
import numpy as np
import pandas as pd
import torch


def compute_attr_availability(nlanes_valid, oneway_valid, width_valid, max_valid, min_valid, avg_valid_frac):
    """Per-node fraction of the predicted attributes that are actually
    observed. Each *_valid arg is a boolean (or [0,1]-fractional, for
    avg_speed's 12 time-of-day slots) array of shape (N,)."""
    return np.stack(
        [nlanes_valid, oneway_valid, width_valid, max_valid, min_valid, avg_valid_frac], axis=1
    ).mean(axis=1).astype(np.float32)


def compute_neighbor_availability(edge_index, avail, num_nodes):
    """Average attribute-availability of each node's line-graph neighbours
    (road segments sharing a junction). NaN for nodes with no neighbours."""
    src, dst = edge_index[0], edge_index[1]
    sums = torch.zeros(num_nodes, dtype=avail.dtype, device=avail.device)
    counts = torch.zeros(num_nodes, dtype=avail.dtype, device=avail.device)
    sums.index_add_(0, src, avail[dst])
    counts.index_add_(0, src, torch.ones_like(dst, dtype=avail.dtype))

    neighbor_avail = torch.full((num_nodes,), float("nan"), dtype=avail.dtype, device=avail.device)
    has_nbr = counts > 0
    neighbor_avail[has_nbr] = sums[has_nbr] / counts[has_nbr]
    return neighbor_avail


def evaluate_scarcity_bins(pred, data, global_masks, neighbor_avail, n_bins, metrics_fn):
    """Bin nodes by neighbour attribute-availability (a data-scarcity proxy)
    and report per-attribute metrics within each bin, evaluated on the
    entries in global_masks (typically a fixed held-out evaluation subset).

    metrics_fn(pred, data, masks) -> dict must already be bound to any extra
    args a specific model's metrics function needs (num_highway, mae_scale).
    """
    valid_np = (~torch.isnan(neighbor_avail)).cpu().numpy()
    values = neighbor_avail.cpu().numpy()

    bin_ids = np.full(values.shape[0], -1, dtype=np.int64)
    if valid_np.sum() > 0:
        bin_ids[valid_np] = pd.qcut(values[valid_np], n_bins, labels=False, duplicates="drop")

    n_actual_bins = int(bin_ids.max()) + 1 if bin_ids.max() >= 0 else 0
    rows = []
    for b in range(n_actual_bins):
        in_bin = torch.from_numpy(bin_ids == b).to(neighbor_avail.device)
        # "avg" masks are (n, 12) — one flag per time-of-day slot — so broadcast
        # the per-node bin membership across that extra dimension.
        bin_masks = {
            k: (v & in_bin.unsqueeze(1)) if v.dim() > 1 else (v & in_bin)
            for k, v in global_masks.items()
        }
        metrics = metrics_fn(pred, data, bin_masks)
        bin_values = values[bin_ids == b]
        rows.append({
            "bin": b,
            "n_nodes": int(in_bin.sum().item()),
            "neighbor_avail_min": float(bin_values.min()),
            "neighbor_avail_max": float(bin_values.max()),
            **metrics,
        })
    return pd.DataFrame(rows)


def plot_scarcity_bins(df, path_prefix, metric_titles=None):
    """One bar chart per attribute metric across data-scarcity bins."""
    import matplotlib.pyplot as plt

    exclude = {"bin", "n_nodes", "neighbor_avail_min", "neighbor_avail_max"}
    if metric_titles is None:
        metric_titles = {c: c for c in df.columns if c not in exclude}

    labels = [f"bin {int(b)}\n[{lo:.2f}, {hi:.2f}]" for b, lo, hi in
              zip(df["bin"], df["neighbor_avail_min"], df["neighbor_avail_max"])]

    for m, title in metric_titles.items():
        plt.figure()
        plt.bar(labels, df[m])
        plt.xlabel("Neighbour attribute-availability bin (scarce → rich)")
        plt.ylabel(m)
        plt.title(f"{title} vs. neighbour data scarcity")
        plt.tight_layout()
        plt.savefig(path_prefix + f"scarcity_{m}.png", format="png")
        plt.close()
