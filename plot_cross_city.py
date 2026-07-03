"""
plot_cross_city.py  –  Plot cross-city evaluation matrices.

Reads all  results/<source>_2_<target>_eval_results.json  files and produces
one heatmap per metric / loss, saved to ./cross-city-results/.

Usage
-----
    python plot_cross_city.py \
        --results_dir ./results \
        --output_dir  ./cross-city-results
for finetuning
    python plot_cross_city.py \
        --results_dir ./results/finetuned \
        --output_dir  ./cross-city-results/finetuned
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    data   : dict[(source, target)] -> json dict
    cities : sorted list of city names (union of all sources + targets)
    """
    data = {}
    city_set = set()

    for fname in os.listdir(results_dir):
        m = FILE_RE.match(fname)
        if not m:
            continue
        source, target = m.group(1), m.group(2)
        fpath = os.path.join(results_dir, fname)
        with open(fpath) as f:
            data[(source, target)] = json.load(f)
        city_set.add(source)
        city_set.add(target)

    cities = sorted(city_set)
    cities = ['jakarta', 'singapore', 'chicago', 'NewYorkCity', 'sanFrancisco', 'washingtonDC']
    return data, cities


def build_matrix(data: dict, cities: list, dotpath: str):
    """
    Build an (N, N) float matrix where entry [i, j] = value for
    source=cities[i], target=cities[j].  NaN if missing.
    """
    N = len(cities)
    mat = np.full((N, N), np.nan)
    for i, src in enumerate(cities):
        for j, tgt in enumerate(cities):
            val = get_nested(data.get((src, tgt)), dotpath)
            if val is not None:
                mat[i, j] = float(val)
    return mat


def plot_matrix(mat, cities, title, cmap, lower_is_better, out_path):
    N = len(cities)
    fig, ax = plt.subplots(figsize=(max(6, N * 1.1), max(5, N * 1.0)))

    # Mask NaN so they show as grey
    masked = np.ma.masked_invalid(mat)
    current_cmap = plt.get_cmap(cmap).copy()
    current_cmap.set_bad(color="#cccccc")

    # Normalise ignoring NaN
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(masked, cmap=current_cmap, norm=norm, aspect="auto")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=current_cmap),
                        ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    arrow = "↓ lower is better" if lower_is_better else "↑ higher is better"
    cbar.set_label(arrow, fontsize=9, labelpad=8)

    # ── Axes labels ───────────────────────────────────────────────────────────
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(cities, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(cities, fontsize=10)
    ax.set_xlabel("Target city  (evaluated on)", fontsize=11, labelpad=8)
    ax.set_ylabel("Source city  (model trained on)", fontsize=11, labelpad=8)

    # ── Cell annotations ──────────────────────────────────────────────────────
    for i in range(N):
        for j in range(N):
            val = mat[i, j]
            if np.isnan(val):
                txt = "N/A"
                color = "#555555"
            else:
                # Pick white or black text based on cell luminance
                normed = (val - vmin) / (vmax - vmin + 1e-9)
                r, g, b, _ = current_cmap(normed)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                color = "white" if luminance < 0.5 else "black"
                txt = f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")

    # ── Diagonal marker ───────────────────────────────────────────────────────
    for k in range(N):
        ax.add_patch(plt.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False, edgecolor="steelblue", linewidth=2.0
        ))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir",  default="./cross-city-results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    data, cities = load_results(args.results_dir)

    if not data:
        print("No result files found. Expected pattern: <source>_2_<target>_eval_results.json")
        return

    print(f"Cities found: {cities}")
    print(f"Pairs loaded: {len(data)}\n")

    for dotpath, label, cmap, lower_is_better in QUANTITIES:
        mat = build_matrix(data, cities, dotpath)

        if np.all(np.isnan(mat)):
            print(f"  Skipping '{label}' — all values missing")
            continue

        # Filename: replace dots and spaces with underscores
        fname = dotpath.replace(".", "_").replace(" ", "_") + ".png"
        out_path = os.path.join(args.output_dir, fname)

        plot_matrix(
            mat      = mat,
            cities   = cities,
            title    = f"Cross-City Evaluation\n{label}",
            cmap     = cmap,
            lower_is_better = lower_is_better,
            out_path = out_path,
        )

    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()