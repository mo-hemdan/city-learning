"""
plot_selective_cross_city.py  –  Plot cross-city evaluation matrices for a
sampled subset of regions per city.

plot_cross_city.py plots every source/target key found in the results
directory. When targets are grid-tile-level (e.g. jakarta has ~350 regions
plus singapore, chicago, ... each with dozens more), that heatmap becomes
too large to read. This script instead samples a fraction of each city's
regions (default 5%, at least --min_per_city of them) and plots only those,
so every city is still represented without drawing hundreds of rows/columns.

Regions are sampled once (grouped by city) from the union of all source and
target keys found in the results directory, then that same sampled set is
used for both the row (source) and column (target) axis, so every figure is
a square matrix over the same regions — matching plot_cross_city.py's
square layout, just over a per-city sample instead of every region.

Reads all  results/<source>_2_<target>_eval_results.json  files and produces
one heatmap per metric / loss, saved to ./cross-city-results-selective/.

The same sampled regions are then reused to produce sampled versions of the
graph2vec embedding-space figures (similarity_matrix.png, pca_scatter.png,
tsne_scatter.png), read from --embeddings_csv (the "named" embeddings CSV
produced by embedding_models/graph2vec_support.py's postprocess step, indexed
by region key). This keeps every figure this script produces — eval heatmaps
and embedding plots alike — restricted to the exact same set of regions.

Usage
-----
    python plot_selective_cross_city.py \
        --results_dir    ./results \
        --embeddings_csv ./embedding_models/data/graph2vec/output/embeddings_named_named.csv \
        --output_dir     ./cross-city-results-selective \
        --sample_frac    0.05 \
        --min_per_city   1 \
        --seed           42
for finetuning
    python plot_selective_cross_city.py \
        --results_dir ./results/finetuned \
        --output_dir  ./cross-city-results-selective/finetuned
"""

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics.pairwise import cosine_similarity

from embedding_models.graph2vec_support import plot_similarity_matrix, plot_scatter


# ── Config ────────────────────────────────────────────────────────────────────

# For each quantity: (json_path, display_label, colormap, lower_is_better)
# json_path uses dot notation to navigate nested dicts, e.g. "losses.hwy"
QUANTITIES = [
    # losses
    ("total_loss",      "Total Loss",            "viridis_r",   True),
    ("losses.hwy",      "Highway Loss (CE)",      "viridis_r",   True),
    ("losses.lan",      "Lanes Loss (CE)",         "viridis_r",   True),
    ("losses.onw",      "Oneway Loss (BCE)",        "viridis_r",   True),
    ("losses.wid",      "Width Loss (Huber)",       "viridis_r",   True),
    ("losses.max",      "Max Speed Loss (Huber)",   "viridis_r",   True),
    ("losses.min",      "Min Speed Loss (Huber)",   "viridis_r",   True),
    ("losses.avg",      "Avg Speed Loss (Huber)",   "viridis_r",   True),
    # metrics
    ("metrics.hwy_macro_f1", "Highway Macro-F1",   "viridis",    False),
    ("metrics.lan_macro_f1", "Lanes Macro-F1",      "viridis",    False),
    ("metrics.onw_auroc",    "Oneway AUROC",         "viridis",    False),
    ("metrics.wid_mae_m",    "Width MAE (m)",        "viridis_r",   True),
    ("metrics.max_mae",      "Max Speed MAE",        "viridis_r",   True),
    ("metrics.min_mae",      "Min Speed MAE",        "viridis_r",   True),
    ("metrics.avg_mae",      "Avg Speed MAE",        "viridis_r",   True),
]

FILE_RE = re.compile(r"^(.+)_2_(.+)_eval_results\.json$")
TILE_RE = re.compile(r"^(.+)_r\d+_c\d+$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def base_city(key: str) -> str:
    """Strip a '_r{row}_c{col}' grid-tile suffix, if present, to get the city name."""
    m = TILE_RE.match(key)
    return m.group(1) if m else key


def get_nested(d: dict, dotpath: str):
    """Navigate a nested dict with dot-notation key, e.g. 'losses.hwy'."""
    for key in dotpath.split("."):
        if d is None:
            return None
        d = d.get(key)
    return d


def load_results(results_dir: str):
    """
    Returns
    -------
    data    : dict[(source, target)] -> json dict
    sources : sorted list of all source keys seen
    targets : sorted list of all target keys seen
    """
    data = {}
    sources, targets = set(), set()

    for fname in os.listdir(results_dir):
        m = FILE_RE.match(fname)
        if not m:
            continue
        source, target = m.group(1), m.group(2)
        fpath = os.path.join(results_dir, fname)
        with open(fpath) as f:
            data[(source, target)] = json.load(f)
        sources.add(source)
        targets.add(target)

    return data, sorted(sources), sorted(targets)


def sample_by_city(keys: list, frac: float, min_per_city: int, seed: int) -> list:
    """
    Group keys by their base city and randomly sample `frac` of each group
    (rounded up), keeping at least `min_per_city` (capped at the group size).
    Returns keys sorted by (city, key) so same-city regions stay contiguous.
    """
    groups = defaultdict(list)
    for k in keys:
        groups[base_city(k)].append(k)

    rng = random.Random(seed)
    sampled = []
    for city, members in groups.items():
        members = sorted(members)
        n = min(len(members), max(min_per_city, math.ceil(len(members) * frac)))
        sampled.extend(rng.sample(members, n))

    return sorted(sampled, key=lambda k: (base_city(k), k))


def build_matrix(data: dict, row_keys: list, col_keys: list, dotpath: str):
    """Build a (len(row_keys), len(col_keys)) float matrix. NaN if missing."""
    mat = np.full((len(row_keys), len(col_keys)), np.nan)
    for i, src in enumerate(row_keys):
        for j, tgt in enumerate(col_keys):
            val = get_nested(data.get((src, tgt)), dotpath)
            if val is not None:
                mat[i, j] = float(val)
    return mat


def city_block_boundaries(keys: list) -> list:
    """Index positions (in `keys`, 0-based) where the base city changes."""
    boundaries = []
    for i in range(1, len(keys)):
        if base_city(keys[i]) != base_city(keys[i - 1]):
            boundaries.append(i)
    return boundaries


def plot_selective_embeddings(embeddings_csv: str, sampled_keys: list, output_dir: str):
    """
    Restrict the graph2vec embedding-space figures (similarity matrix, PCA
    and t-SNE scatter) to `sampled_keys`, reusing the same plotting code as
    embedding_models/graph2vec_support.py's postprocess step.
    """
    if not os.path.exists(embeddings_csv):
        print(f"  Skipping embedding plots — {embeddings_csv} not found")
        return

    df = pd.read_csv(embeddings_csv, index_col=0)

    keys = [k for k in sampled_keys if k in df.index]
    missing = [k for k in sampled_keys if k not in df.index]
    if missing:
        print(f"  Note: {len(missing)} sampled region(s) have no embedding, skipping: {missing}")
    if len(keys) < 2:
        print(f"  Skipping embedding plots — only {len(keys)} sampled region(s) have embeddings")
        return

    embeddings = df.loc[keys].values.astype(np.float32)
    sim_matrix = cosine_similarity(embeddings)

    print(f"\nGenerating embedding plots for {len(keys)} sampled regions …")
    plot_similarity_matrix(sim_matrix, keys, os.path.join(output_dir, "similarity_matrix.png"))
    plot_scatter(embeddings, keys, os.path.join(output_dir, "pca_scatter.png"), method="pca")
    plot_scatter(embeddings, keys, os.path.join(output_dir, "tsne_scatter.png"), method="tsne")


def plot_matrix(mat, row_keys, col_keys, title, cmap, lower_is_better, out_path):
    n_rows, n_cols = len(row_keys), len(col_keys)
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.55), max(5, n_rows * 0.5)))

    # Mask NaN so they show as grey
    masked = np.ma.masked_invalid(mat)
    current_cmap = plt.get_cmap(cmap).copy()
    current_cmap.set_bad(color="#cccccc")

    # Normalise ignoring NaN
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    norm = Normalize(vmin=vmin, vmax=vmax)

    ax.imshow(masked, cmap=current_cmap, norm=norm, aspect="auto")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=current_cmap),
                        ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    arrow = "↓ lower is better" if lower_is_better else "↑ higher is better"
    cbar.set_label(arrow, fontsize=9, labelpad=8)

    # ── Axes labels ───────────────────────────────────────────────────────────
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_keys, rotation=90, ha="center", fontsize=7)
    ax.set_yticklabels(row_keys, fontsize=7)
    ax.set_xlabel("Target region  (evaluated on, sampled per city)", fontsize=11, labelpad=8)
    ax.set_ylabel("Source region  (model trained on, same sampled regions)", fontsize=11, labelpad=8)

    # ── Cell annotations (only if the grid is small enough to stay readable) ──
    annotate = n_rows * n_cols <= 400
    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                normed = (val - vmin) / (vmax - vmin + 1e-9)
                r, g, b, _ = current_cmap(normed)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                color = "white" if luminance < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color, fontweight="bold")

    # ── Same-region marker (row key == col key) ─────────────────────────────
    col_pos = {k: j for j, k in enumerate(col_keys)}
    for i, k in enumerate(row_keys):
        if k in col_pos:
            ax.add_patch(plt.Rectangle(
                (col_pos[k] - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="steelblue", linewidth=1.5
            ))

    # ── City block separators ────────────────────────────────────────────────
    for b in city_block_boundaries(col_keys):
        ax.axvline(b - 0.5, color="black", linewidth=1.0)
    for b in city_block_boundaries(row_keys):
        ax.axhline(b - 0.5, color="black", linewidth=1.0)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",  default="./results")
    parser.add_argument("--embeddings_csv", default="./embedding_models/data/graph2vec/output/embeddings_named_named.csv",
                        help="Named graph2vec embeddings CSV (index = region key) used for "
                             "similarity_matrix.png / pca_scatter.png / tsne_scatter.png")
    parser.add_argument("--output_dir",   default="./cross-city-results-selective")
    parser.add_argument("--sample_frac",  type=float, default=0.05,
                        help="Fraction of each city's regions to sample (default 5%%)")
    parser.add_argument("--min_per_city", type=int,   default=1,
                        help="Minimum regions kept per city, even if sample_frac rounds to 0")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed for sampling (for reproducible plots)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    data, all_sources, all_targets = load_results(args.results_dir)

    if not data:
        print("No result files found. Expected pattern: <source>_2_<target>_eval_results.json")
        return

    all_keys = sorted(set(all_sources) | set(all_targets))
    sampled_keys = sample_by_city(all_keys, args.sample_frac, args.min_per_city, args.seed)
    row_keys = col_keys = sampled_keys

    print(f"Regions: {len(all_keys)} total -> {len(sampled_keys)} sampled")
    print(f"Sampled regions (used as both source and target): {sampled_keys}\n")

    for dotpath, label, cmap, lower_is_better in QUANTITIES:
        mat = build_matrix(data, row_keys, col_keys, dotpath)

        if np.all(np.isnan(mat)):
            print(f"  Skipping '{label}' — all sampled values missing")
            continue

        fname = dotpath.replace(".", "_").replace(" ", "_") + ".png"
        out_path = os.path.join(args.output_dir, fname)

        plot_matrix(
            mat      = mat,
            row_keys = row_keys,
            col_keys = col_keys,
            title    = f"Cross-City Evaluation (sampled {args.sample_frac:.0%} per city)\n{label}",
            cmap     = cmap,
            lower_is_better = lower_is_better,
            out_path = out_path,
        )

    plot_selective_embeddings(args.embeddings_csv, sampled_keys, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
