"""
plot_genome_similarity_correlation.py – Correlate graph2vec genome similarity
with cross-city transfer performance.

For every (source, target) pair with source != target, plots the graph2vec
cosine similarity between the two cities' genomes against the transfer
metric/loss obtained when a model trained on `source` is evaluated on
`target`. One scatter plot per quantity is produced, styled to match the
heatmaps from plot_cross_city.py (same viridis colormaps, same "lower/higher
is better" colorbar, same fonts/dpi).

Reads:
    - results/<source>_2_<target>_eval_results.json   (transfer metrics)
    - embedding_models/data/graph2vec/output/similarity_matrix.csv
      (genome cosine similarity between cities)

Usage
-----
    python plot_genome_similarity_correlation.py \
        --results_dir    ./results \
        --similarity_csv ./embedding_models/data/graph2vec/output/similarity_matrix.csv \
        --output_dir     ./cross-city-results/genome-similarity-correlation
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import pearsonr, spearmanr

from plot_cross_city import QUANTITIES, get_nested, load_results

FILE_RE = re.compile(r"^(.+)_2_(.+)_eval_results\.json$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_similarity(csv_path: str) -> pd.DataFrame:
    """Load the cosine-similarity matrix produced by graph2vec_support.py."""
    return pd.read_csv(csv_path, index_col=0)


def collect_points(data: dict, sim_df: pd.DataFrame, dotpath: str, include_diagonal: bool):
    """
    Returns lists (sim, val, src, tgt) for every (source, target) pair that
    has both a genome-similarity entry and a transfer-metric value.
    """
    sims, vals, srcs, tgts = [], [], [], []
    for (src, tgt), payload in data.items():
        if not include_diagonal and src == tgt:
            continue
        if src not in sim_df.index or tgt not in sim_df.columns:
            continue
        val = get_nested(payload, dotpath)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        sims.append(float(sim_df.loc[src, tgt]))
        vals.append(float(val))
        srcs.append(src)
        tgts.append(tgt)
    return np.array(sims), np.array(vals), srcs, tgts


def plot_correlation(sims, vals, srcs, tgts, label, cmap, lower_is_better, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    current_cmap = plt.get_cmap(cmap).copy()
    vmin, vmax = np.min(vals), np.max(vals)
    norm = Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        sims, vals, c=vals, cmap=current_cmap, norm=norm,
        s=90, edgecolor="white", linewidth=0.8, zorder=3,
    )

    # ── Regression line (steelblue, matching the diagonal-marker accent) ──────
    if len(sims) >= 2 and np.ptp(sims) > 0:
        m, b = np.polyfit(sims, vals, 1)
        xs = np.linspace(sims.min(), sims.max(), 100)
        ax.plot(xs, m * xs + b, color="steelblue", linewidth=2.0,
                 linestyle="--", zorder=2, label="linear fit")

    # ── Colorbar (same convention as the heatmaps) ─────────────────────────────
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=current_cmap),
                         ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    arrow = "↓ lower is better" if lower_is_better else "↑ higher is better"
    cbar.set_label(arrow, fontsize=9, labelpad=8)

    # ── Correlation stats ───────────────────────────────────────────────────
    r, p = pearsonr(sims, vals)
    rho, p_rho = spearmanr(sims, vals)
    stats_txt = (
        f"Pearson r = {r:+.3f}  (p = {p:.3g})\n"
        f"Spearman ρ = {rho:+.3f}  (p = {p_rho:.3g})\n"
        f"n = {len(sims)} pairs"
    )
    ax.text(
        0.03, 0.97, stats_txt, transform=ax.transAxes,
        fontsize=9.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel("Genome Cosine Similarity (source, target)", fontsize=11, labelpad=8)
    ax.set_ylabel(label, fontsize=11, labelpad=8)
    ax.set_title(f"Genome Similarity vs. {label}\n(cross-city transfer, source ≠ target)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(labelsize=9.5)
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")
    return r, p, rho, p_rho


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--similarity_csv",
                         default="./embedding_models/data/graph2vec/output/similarity_matrix.csv")
    parser.add_argument("--output_dir",
                         default="./cross-city-results/genome-similarity-correlation")
    parser.add_argument("--include_diagonal", action="store_true",
                         help="Include source==target (self-transfer) pairs, "
                              "which are trivially similarity=1.0.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from:   {args.results_dir}")
    data, cities = load_results(args.results_dir)
    if not data:
        print("No result files found. Expected pattern: <source>_2_<target>_eval_results.json")
        return

    print(f"Loading similarity from: {args.similarity_csv}")
    sim_df = load_similarity(args.similarity_csv)

    summary = []
    for dotpath, label, cmap, lower_is_better in QUANTITIES:
        sims, vals, srcs, tgts = collect_points(
            data, sim_df, dotpath, args.include_diagonal
        )
        if len(sims) < 2:
            print(f"  Skipping '{label}' — fewer than 2 valid pairs")
            continue

        fname = "corr_" + dotpath.replace(".", "_").replace(" ", "_") + ".png"
        out_path = os.path.join(args.output_dir, fname)

        r, p, rho, p_rho = plot_correlation(
            sims, vals, srcs, tgts, label, cmap, lower_is_better, out_path
        )
        summary.append((label, r, p, rho, p_rho, len(sims)))

    print("\n── Summary ──────────────────────────────────────────────")
    print(f"{'Metric':<24}{'Pearson r':>12}{'p-value':>12}{'Spearman rho':>14}{'n':>6}")
    for label, r, p, rho, p_rho, n in summary:
        print(f"{label:<24}{r:>12.3f}{p:>12.3g}{rho:>14.3f}{n:>6}")

    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
