"""
prepare_graph2vec_input.py

Converts city edge parquets + speed matrices into the JSON format expected by
benedekrozemberczki/graph2vec.

Two modes for node features:
  --feature_mode structural   degree bucket only (default)
  --feature_mode attributed   road metadata + avg_speed discretised into bins,
                              combined into one composite integer label per node

Usage
-----
    # Step 1: prepare input
    python prepare_graph2vec_input.py \
        --mode         prepare \
        --feature_mode attributed \
        --data_dir     ./data/raw_data \
        --cities_json  ./cities.json \
        --input_dir    ./data/graph2vec/input

    # Step 2: run graph2vec
    python graph2vec/src/graph2vec.py \
        --input-path  ./data/graph2vec/input \
        --output-path ./data/graph2vec/output/embeddings.csv \
        --dimensions 128 --wl-iterations 3 --epochs 50 --workers 4

    # Step 3: postprocess → city names + plots
    python prepare_graph2vec_input.py \
        --mode        postprocess \
        --cities_json ./cities.json \
        --input_dir   ./data/graph2vec/input \
        --output_csv  ./data/graph2vec/output/embeddings.csv \
        --plot_dir    ./graph_embeddings
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.expanduser("~/websites/mapedia"))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage

from modules.city_learning.src.processing import (
    build_line_graph_edge_index,
    aggregate_speed_matrix,
    nlanes_to_class,
    oneway_to_class,
    highway_to_class,
)

N_BINS = 8   # number of quantile bins for continuous features


# ── Feature builders ──────────────────────────────────────────────────────────

def _quantile_bin(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bin a 1-D array of floats into integer labels 0..n_bins-1.
    NaN values are assigned to bin 0."""
    out = np.zeros(len(values), dtype=np.int32)
    valid = ~np.isnan(values)
    if valid.sum() > 1:
        bins = np.quantile(values[valid], np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        out[valid] = np.digitize(values[valid], bins[1:-1]).astype(np.int32)
    return out


def structural_features(edge_list: list, N: int) -> dict:
    """Degree-bucket label per node (structural only)."""
    deg = Counter()
    for s, t in edge_list:
        deg[s] += 1
        deg[t] += 1
    degrees = np.array([deg.get(i, 0) for i in range(N)], dtype=np.float32)
    bins    = np.quantile(degrees, np.linspace(0, 1, N_BINS + 1))
    bins    = np.unique(bins)
    buckets = np.digitize(degrees, bins[1:-1]).astype(np.int32)
    return {str(i): int(buckets[i]) for i in range(N)}


def attributed_features(edges: gpd.GeoDataFrame,
                         speed_matrix: np.ndarray,
                         edge_list: list) -> dict:
    """
    Composite integer label per node combining:
      - degree bucket          (0..N_BINS-1)
      - nlanes class           (0,1,2 or 0 for missing)
      - oneway                 (0,1 or 0 for missing)
      - highway type id        (int)
      - width bin              (0..N_BINS-1)
      - max_speed bin          (0..N_BINS-1)
      - min_speed bin          (0..N_BINS-1)
      - avg_speed (mean) bin   (0..N_BINS-1)

    Each feature is assigned a prime-based stride so the composite label is
    unique per combination and still fits in a Python int.
    """
    N = len(edges)

    # ── Degree ────────────────────────────────────────────────────────────────
    deg = Counter()
    for s, t in edge_list:
        deg[s] += 1
        deg[t] += 1
    degrees = np.array([deg.get(i, 0) for i in range(N)], dtype=np.float32)
    deg_bin = _quantile_bin(degrees)

    # ── Categorical ───────────────────────────────────────────────────────────
    nlanes  = nlanes_to_class(edges["nlanes"])
    nlanes  = np.where(nlanes == -1, 0, nlanes).astype(np.int32)     # missing → 0

    oneway_raw = oneway_to_class(edges["oneway"]).to_numpy(dtype=np.float32)
    oneway  = np.where(np.isnan(oneway_raw), 0, oneway_raw).astype(np.int32)

    hwy_ids, _, _, _, _, _, _ = highway_to_class(edges["road_type"])
    hwy     = np.array(hwy_ids, dtype=np.int32)

    # ── Continuous → bins ─────────────────────────────────────────────────────
    width_bin = _quantile_bin(edges["width"].to_numpy(dtype=np.float32))
    max_bin   = _quantile_bin(edges["max_speed"].to_numpy(dtype=np.float32))
    min_bin   = _quantile_bin(edges["min_speed"].to_numpy(dtype=np.float32))

    # avg_speed: aggregate to one mean per road (ignoring NaN)
    # speed_matrix shape after aggregate_speed_matrix: (N, 2, 6) → flatten → (N, 12)
    avg_flat  = speed_matrix.reshape(N, -1).astype(np.float32)
    avg_mean  = np.nanmean(avg_flat, axis=1)                          # (N,) NaN if all missing
    avg_bin   = _quantile_bin(avg_mean)

    # ── Composite label via mixed-radix encoding ───────────────────────────────
    # Strides chosen so each feature occupies its own "digit" in a big integer.
    # This keeps labels unique per combination without hashing collisions.
    n_hwy    = int(hwy.max()) + 1
    strides  = [1,
                N_BINS,                  # after deg_bin
                N_BINS * 3,              # after nlanes  (3 classes)
                N_BINS * 3 * 2,          # after oneway  (2 classes)
                N_BINS * 3 * 2 * n_hwy, # after hwy
                N_BINS * 3 * 2 * n_hwy * N_BINS,
                N_BINS * 3 * 2 * n_hwy * N_BINS ** 2,
                N_BINS * 3 * 2 * n_hwy * N_BINS ** 3,
                ]

    composite = (
        deg_bin   * strides[0] +
        nlanes    * strides[1] +
        oneway    * strides[2] +
        hwy       * strides[3] +
        width_bin * strides[4] +
        max_bin   * strides[5] +
        min_bin   * strides[6] +
        avg_bin   * strides[7]
    )

    return {str(i): int(composite[i]) for i in range(N)}


# ── Prepare ───────────────────────────────────────────────────────────────────

def prepare(data_dir, cities_json, input_dir, feature_mode):
    os.makedirs(input_dir, exist_ok=True)

    with open(cities_json) as f:
        cities = list(json.load(f).keys())

    print(f"Feature mode: {feature_mode}\n")
    index_map = {}

    for idx, city in enumerate(cities):
        edges_path = os.path.join(data_dir, f"{city}_edges.parquet")
        speed_path = os.path.join(data_dir, f"{city}_speed_matrix.npy")

        if not os.path.exists(edges_path):
            print(f"  ⚠ Skipping {city} — edges file not found")
            continue

        print(f"Processing {city} (index={idx}) …")
        edges = gpd.read_parquet(edges_path)
        edges = edges.reset_index().rename(columns={"index": "idx"})
        N = len(edges)

        edge_index = build_line_graph_edge_index(
            edges, u_col="source", v_col="target", eid_col="idx"
        ).cpu().numpy()
        edge_list = edge_index.T.tolist()
        print(f"  {N:,} nodes | {len(edge_list):,} edges")

        if feature_mode == "structural":
            features = structural_features(edge_list, N)

        else:  # attributed
            if not os.path.exists(speed_path):
                print(f"  ⚠ Speed matrix not found, falling back to structural for {city}")
                features = structural_features(edge_list, N)
            else:
                speed_matrix = np.load(speed_path)
                speed_matrix = aggregate_speed_matrix(
                    speed_matrix,
                    ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
                )
                features = attributed_features(edges, speed_matrix, edge_list)

        graph_json = {"edges": edge_list, "features": features}

        out_path = os.path.join(input_dir, f"{idx}.json")
        with open(out_path, "w") as f:
            json.dump(graph_json, f)
        print(f"  Saved → {out_path}")

        index_map[str(idx)] = city

    # Save index map one level ABOVE input_dir so graph2vec doesn't parse it
    map_path = os.path.join(os.path.dirname(input_dir), "_index_map.json")
    with open(map_path, "w") as f:
        json.dump(index_map, f, indent=2)
    print(f"\nIndex map saved → {map_path}")
    print("\nNow run graph2vec:")
    print(f"  python graph2vec/src/graph2vec.py \\")
    print(f"      --input-path  {input_dir} \\")
    print(f"      --output-path ./data/graph2vec/output/embeddings.csv \\")
    print(f"      --dimensions 128 --wl-iterations 3 --epochs 50 --workers 4")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_similarity_matrix(sim_matrix, cities, out_path, font_scale=1.0):
    N   = len(cities)
    fig, ax = plt.subplots(figsize=(max(6, N * 1.1), max(5, N * 1.0)))

    masked = np.ma.masked_invalid(sim_matrix)
    cmap   = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#cccccc")

    vmin = np.nanmin(sim_matrix)
    vmax = np.nanmax(sim_matrix)
    norm = Normalize(vmin=vmin, vmax=vmax)

    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("↑ higher is better", fontsize=9, labelpad=8)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(cities, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(cities, fontsize=10)
    ax.set_xlabel("City", fontsize=11, labelpad=8)
    ax.set_ylabel("City", fontsize=11, labelpad=8)

    for i in range(N):
        for j in range(N):
            val = sim_matrix[i, j]
            if np.isnan(val):
                txt, color = "N/A", "#555555"
            else:
                normed = (val - vmin) / (vmax - vmin + 1e-9)
                r, g, b, _ = cmap(normed)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                color = "white" if luminance < 0.5 else "black"
                txt = f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")

    for k in range(N):
        ax.add_patch(plt.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False, edgecolor="steelblue", linewidth=2.0
        ))

    ax.set_title("City Graph Structural Similarity\n(Graph2Vec)",
                 fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_dendrogram(embeddings, cities, out_path):
    dist = np.clip(pdist(embeddings, metric="cosine"), 0, None)
    Z    = linkage(dist, method="ward")
    fig, ax = plt.subplots(figsize=(max(6, len(cities) * 1.1), 4))
    dendrogram(Z, labels=cities, ax=ax, leaf_rotation=35, leaf_font_size=14,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("City Structural Similarity — Dendrogram", fontsize=16, fontweight="bold")
    ax.set_ylabel("Distance", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_scatter(embeddings, cities, out_path, method="pca"):
    from sklearn.manifold import TSNE
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(cities)))
    fig, ax = plt.subplots(figsize=(6, 5))

    if method == "pca":
        pca    = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        xlabel = f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        ylabel = f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
        title  = "City Graph Embeddings — PCA Projection"
    else:
        perplexity = min(5, len(cities) - 1)
        coords = TSNE(n_components=2, perplexity=perplexity,
                      random_state=42, n_iter=1000, init="pca").fit_transform(embeddings)
        xlabel = "Dim 1"
        ylabel = "Dim 2"
        title  = f"City Graph Embeddings — t-SNE (perplexity={perplexity})"

    for i, (city, c) in enumerate(zip(cities, colors)):
        ax.scatter(coords[i, 0], coords[i, 1], s=220, color=c, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(city, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(9, 5), fontsize=13,
                    fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Postprocess ───────────────────────────────────────────────────────────────

def postprocess(cities_json, output_csv, plot_dir, input_dir):
    os.makedirs(plot_dir, exist_ok=True)

    map_path = os.path.join(os.path.dirname(input_dir), "_index_map.json")
    with open(map_path) as f:
        index_map = json.load(f)

    df = pd.read_csv(output_csv)
    if "type" in df.columns:
        df = df.set_index("type")
    df.index = [index_map.get(str(i), str(i)) for i in df.index]
    df.index.name = "city"

    named_path = output_csv.replace(".csv", "_named.csv")
    df.to_csv(named_path)
    print(f"Named embeddings saved → {named_path}")

    cities     = list(df.index)
    embeddings = df.values.astype(np.float32)
    sim_matrix = cosine_similarity(embeddings)

    pd.DataFrame(sim_matrix, index=cities, columns=cities).to_csv(
        os.path.join(plot_dir, "similarity_matrix.csv")
    )

    print("\nGenerating plots …")
    plot_similarity_matrix(sim_matrix, cities,
                           os.path.join(plot_dir, "similarity_matrix.png"))
    plot_dendrogram(embeddings, cities,
                    os.path.join(plot_dir, "dendrogram.png"))
    plot_scatter(embeddings, cities,
                 os.path.join(plot_dir, "pca_scatter.png"), method="pca")
    plot_scatter(embeddings, cities,
                 os.path.join(plot_dir, "tsne_scatter.png"), method="tsne")

    print("\n=== Most similar city pairs ===")
    n = len(cities)
    pairs = [(sim_matrix[i, j], cities[i], cities[j])
             for i in range(n) for j in range(i + 1, n)]
    for sim, c1, c2 in sorted(pairs, reverse=True):
        print(f"  {c1:<16} ↔  {c2:<16}  similarity = {sim:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         choices=["prepare", "postprocess"], required=True)
    parser.add_argument("--feature_mode", choices=["structural", "attributed"],
                        default="structural",
                        help="'structural' = degree only; 'attributed' = road metadata + speed")
    parser.add_argument("--data_dir",     default="./data/raw_data")
    parser.add_argument("--cities_json",  default="./cities.json")
    parser.add_argument("--input_dir",    default="./data/graph2vec/input")
    parser.add_argument("--output_csv",   default="./data/graph2vec/output/embeddings.csv")
    parser.add_argument("--plot_dir",     default="./graph_embeddings")
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare(args.data_dir, args.cities_json, args.input_dir, args.feature_mode)
    else:
        postprocess(args.cities_json, args.output_csv, args.plot_dir, args.input_dir)


if __name__ == "__main__":
    main()