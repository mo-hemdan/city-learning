"""
netlsd_support.py

Computes NetLSD spectral graph descriptors for city road networks — the same
road-segment line graph (nodes = segments, edges = segments sharing an
endpoint) that graph2vec_support.py builds — and generates the same
similarity/PCA/t-SNE plots.

Unlike graph2vec, NetLSD needs no training: given a graph, netlsd.heat()/
wave() directly returns a fixed-length descriptor from the graph Laplacian's
eigenvalue spectrum. It is purely structural (topology only) — it does not
use road attributes like highway type, width or speed.

Usage
-----
    # Step 1: compute embeddings
    python netlsd_support.py \
        --mode        compute \
        --data_dir    ./data/raw_data \
        --cities_json ./cities.json \
        --output_csv  ./embedding_models/data/netlsd/output/embeddings_named.csv \
        --kernel      heat \
        --dimensions  128

    # Step 2: postprocess -> plots
    python netlsd_support.py \
        --mode        postprocess \
        --output_csv  ./embedding_models/data/netlsd/output/embeddings_named.csv \
        --plot_dir    ./embedding_models/data/netlsd/output
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.expanduser("~/websites/mapedia"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NetLSD"))

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import netlsd
from sklearn.metrics.pairwise import cosine_similarity

from modules.city_learning.src.processing import build_line_graph_edge_index
from graph2vec_support import plot_similarity_matrix, plot_dendrogram, plot_scatter


# ── Graph construction ──────────────────────────────────────────────────────

def edges_to_networkx(edges: gpd.GeoDataFrame) -> nx.Graph:
    """
    Road-segment line graph as an undirected networkx Graph: nodes = road
    segments, edges = segment pairs sharing an endpoint. Matches the node/edge
    definition graph2vec_support.py uses, so descriptors stay comparable.
    """
    edges = edges.reset_index().rename(columns={"index": "idx"})
    N = len(edges)
    edge_index = build_line_graph_edge_index(
        edges, u_col="source", v_col="target", eid_col="idx"
    ).cpu().numpy()

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edge_index.T.tolist())
    return G


# ── Compute ──────────────────────────────────────────────────────────────────

def compute(data_dir, cities_json, output_csv, kernel, dimensions, normalization):
    with open(cities_json) as f:
        cities = list(json.load(f).keys())

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    timescales = (np.logspace(-2, 2, dimensions) if kernel == "heat"
                  else np.linspace(0, 2 * np.pi, dimensions))
    kernel_fn = netlsd.heat if kernel == "heat" else netlsd.wave

    rows = {}
    for city in cities:
        edges_path = os.path.join(data_dir, f"{city}_edges.parquet")
        if not os.path.exists(edges_path):
            print(f"  ⚠ Skipping {city} — edges file not found")
            continue

        print(f"Processing {city} …")
        edges = gpd.read_parquet(edges_path)
        G = edges_to_networkx(edges)
        print(f"  {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

        descriptor = kernel_fn(G, timescales=timescales, normalization=normalization)
        rows[city] = np.real(descriptor).astype(np.float32)

    if not rows:
        print("[ERROR] No cities processed — check data_dir / cities_json")
        sys.exit(1)

    dim_cols = [f"x_{i}" for i in range(dimensions)]
    df = pd.DataFrame.from_dict(rows, orient="index", columns=dim_cols)
    df.index.name = "city"
    df.to_csv(output_csv)
    print(f"\nEmbeddings saved → {output_csv}  ({df.shape[0]} cities x {df.shape[1]} dims)")


# ── Postprocess ───────────────────────────────────────────────────────────────

def postprocess(output_csv, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.read_csv(output_csv, index_col=0)
    cities     = list(df.index)
    embeddings = df.values.astype(np.float32)
    sim_matrix = cosine_similarity(embeddings)

    pd.DataFrame(sim_matrix, index=cities, columns=cities).to_csv(
        os.path.join(plot_dir, "similarity_matrix.csv")
    )

    print("\nGenerating plots …")
    plot_similarity_matrix(sim_matrix, cities,
                           os.path.join(plot_dir, "similarity_matrix.png"),
                           method_label="NetLSD")
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
    parser.add_argument("--mode",         choices=["compute", "postprocess"], required=True)
    parser.add_argument("--data_dir",     default="./data/raw_data")
    parser.add_argument("--cities_json",  default="./cities.json")
    parser.add_argument("--output_csv",   default="./embedding_models/data/netlsd/output/embeddings_named.csv")
    parser.add_argument("--plot_dir",     default="./embedding_models/data/netlsd/output")
    parser.add_argument("--kernel",       choices=["heat", "wave"], default="heat")
    parser.add_argument("--dimensions",   type=int, default=128,
                        help="Number of timescale samples in the descriptor "
                             "(must match the pgvector column width, 128)")
    parser.add_argument("--normalization", default="empty", choices=["empty", "complete", "none"],
                        help="netlsd trace normalization; 'none' returns the raw kernel trace")
    args = parser.parse_args()

    normalization = None if args.normalization == "none" else args.normalization

    if args.mode == "compute":
        compute(args.data_dir, args.cities_json, args.output_csv,
                args.kernel, args.dimensions, normalization)
    else:
        postprocess(args.output_csv, args.plot_dir)


if __name__ == "__main__":
    main()
