"""
plot_grid_success.py – Visualize which grid-cell training runs succeeded.

For each city, lays out its grid cells (row/col, as defined in
city_grids.json) on a 2D grid and colors each cell by whether a checkpoint
directory exists for it under the checkpoints folder:
  - colored : checkpoints/{city}_r{row}_c{col}/ exists  -> run succeeded
  - white   : the cell is a defined grid tile but has no checkpoint dir
              -> run failed / hasn't produced a checkpoint yet
  - blank   : row/col combination isn't a grid tile for this city at all
              (outside the city's footprint)

Success is determined purely by directory existence (not its contents),
matching how 2_train_on_graphs.py creates the checkpoint dir on first save.

Usage
-----
    python plot_grid_success.py \
        --city_grids_path ./city_grids.json \
        --checkpoints_dir ./checkpoints \
        --output_dir ./plots/grid-success
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

TILE_RE = re.compile(r"^(.+)_r(\d+)_c(\d+)$")


def load_city_tiles(city_grids_path: str):
    """Group grid tiles from city_grids.json by city -> {(row, col): tile_key}."""
    with open(city_grids_path) as f:
        grids = json.load(f)

    cities = {}
    for key in grids:
        m = TILE_RE.match(key)
        if not m:
            continue
        city, row, col = m.group(1), int(m.group(2)), int(m.group(3))
        cities.setdefault(city, {})[(row, col)] = key

    return cities


def plot_city_grid(city: str, tiles: dict, checkpoints_dir: str, out_path: str):
    rows = [r for r, _ in tiles]
    cols = [c for _, c in tiles]
    n_rows, n_cols = max(rows) + 1, max(cols) + 1

    # 0 = not a grid tile, 1 = tile but no checkpoint (failed), 2 = checkpoint exists (success)
    status = [[0] * n_cols for _ in range(n_rows)]
    n_success = 0
    for (row, col), tile_key in tiles.items():
        ckpt_dir = os.path.join(checkpoints_dir, tile_key)
        if os.path.isdir(ckpt_dir):
            status[row][col] = 2
            n_success += 1
        else:
            status[row][col] = 1

    n_tiles = len(tiles)
    cmap = ListedColormap(["#eeeeee", "#ffffff", "#2ca02c"])

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.5), max(5, n_rows * 0.5)))
    ax.imshow(status, cmap=cmap, vmin=0, vmax=2, origin="lower", aspect="equal")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(range(n_cols), fontsize=8)
    ax.set_yticklabels(range(n_rows), fontsize=8)
    ax.set_xlabel("col", fontsize=10)
    ax.set_ylabel("row", fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#999999")
    ax.set_xticks([x - 0.5 for x in range(n_cols + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(n_rows + 1)], minor=True)
    ax.grid(which="minor", color="#999999", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="#999999", label="Succeeded"),
        Patch(facecolor="#ffffff", edgecolor="#999999", label="Failed"),
        Patch(facecolor="#eeeeee", edgecolor="#999999", label="Not a grid tile"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
               fontsize=9, frameon=False)

    ax.set_title(f"{city}: {n_success}/{n_tiles} grid cells succeeded", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {city}: {n_success}/{n_tiles} succeeded -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_grids_path", default="./city_grids.json")
    parser.add_argument("--checkpoints_dir", default="./checkpoints")
    parser.add_argument("--output_dir", default="./plots/grid-success")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cities = load_city_tiles(args.city_grids_path)
    print(f"Found {len(cities)} cities with grid tiles in {args.city_grids_path}")

    for city, tiles in cities.items():
        out_path = os.path.join(args.output_dir, f"{city}_grid_success.png")
        plot_city_grid(city, tiles, args.checkpoints_dir, out_path)

    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
