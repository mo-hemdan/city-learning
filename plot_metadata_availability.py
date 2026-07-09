"""
plot_metadata_availability.py – Plot per-city metadata availability.

For each city's raw edge table, computes the percentage of edges for which
each metadata attribute (road type, number of lanes, oneway, width, max/min
speed, avg speed) is actually present (non-missing), and renders a single
city × metadata-type heatmap (rows = cities, columns = metadata types,
cell value = % available), saved as a PNG.

Usage
-----
    python plot_metadata_availability.py \
        --data_dir   ./data/raw_data \
        --output_dir ./metadata-availability
"""

import argparse
import os

import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

CITIES = ["jakarta", "singapore", "chicago", "NewYorkCity", "sanFrancisco", "washingtonDC"]

# (edges column, display label)
COLUMN_METADATA = [
    ("road_type", "Road Type"),
    ("nlanes",    "# Lanes"),
    ("oneway",    "Oneway"),
    ("width",     "Width"),
    ("max_speed", "Max Speed"),
    ("min_speed", "Min Speed"),
]
AVG_SPEED_LABEL = "Avg Speed"


def avg_speed_availability(data_dir: str, city: str) -> float:
    """% of edges with at least one non-NaN reading anywhere in the speed matrix."""
    path = os.path.join(data_dir, f"{city}_speed_matrix.npy")
    speed_matrix = np.load(path, mmap_mode="r")
    has_reading = ~np.all(np.isnan(speed_matrix), axis=tuple(range(1, speed_matrix.ndim)))
    return float(has_reading.mean() * 100)


def build_availability_matrix(data_dir: str, cities: list):
    labels = [label for _, label in COLUMN_METADATA] + [AVG_SPEED_LABEL]
    mat = np.full((len(cities), len(labels)), np.nan)

    for i, city in enumerate(cities):
        edges = gpd.read_parquet(os.path.join(data_dir, f"{city}_edges.parquet"))
        for j, (col, _) in enumerate(COLUMN_METADATA):
            mat[i, j] = (1 - edges[col].isnull().mean()) * 100
        mat[i, len(COLUMN_METADATA)] = avg_speed_availability(data_dir, city)

    return mat, labels


def plot_availability(mat, cities, labels, out_path):
    mat = mat.T  # rows = metadata types, columns = cities
    n_rows, n_cols = mat.shape
    fig, ax = plt.subplots(figsize=(max(7, n_cols * 1.3), max(5, n_rows * 0.9)))

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=100)

    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Availability (%)", fontsize=9, labelpad=8)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(cities, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("City", fontsize=11, labelpad=8)
    ax.set_ylabel("Metadata type", fontsize=11, labelpad=8)

    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            normed = (val - 0) / 100
            r, g, b, _ = cmap(normed)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Metadata Availability by City", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data/raw_data")
    parser.add_argument("--output_dir", default="./metadata-availability")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading raw edge data from: {args.data_dir}")
    mat, labels = build_availability_matrix(args.data_dir, CITIES)

    for i, city in enumerate(CITIES):
        print(f"  {city}: " + ", ".join(f"{lab}={mat[i, j]:.1f}%" for j, lab in enumerate(labels)))

    out_path = os.path.join(args.output_dir, "metadata_availability.png")
    plot_availability(mat, CITIES, labels, out_path)

    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
