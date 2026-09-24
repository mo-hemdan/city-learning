"""
stats_support.py

Computes a hand-designed "genome" descriptor per city: a fixed-length,
interpretable composite of topological, geometric/morphological, road/speed
attribute-distribution, and scale statistics, computed on each city's road
network in isolation (fully inductive — no cross-city training, unlike
graph2vec or SARN). Optionally appends an off-the-shelf structural component
(NetLSD's heat-kernel trace) as an extra block.

Rationale: none of graph2vec, NetLSD, or SARN individually captures the road
*attribute* distributions (highway-class mix, speed limits, lane counts) that
MultiAttrGAT actually transfers across cities — so this descriptor adds that
block by hand instead of hoping an off-the-shelf graph embedding infers it.

Blocks
------
Topological   — degree histogram (dead-ends / 3-way / 4-way / 5-way+),
                intersection & edge density, average circuity, connectivity
                (largest-component fraction, average clustering).
Geometric     — orientation entropy + grid-order index (Boeing 2019), edge
                length distribution, density gradient from the centroid.
                (Block-area distribution is intentionally omitted: this
                pipeline has no building/parcel polygon data.)
Attribute     — road-class mix, speed-limit / lane-count distributions,
                one-way fraction, and degree<->road-class / degree<->speed
                correlations — the attribute-topology coupling MultiAttrGAT
                relies on.
Scale         — log(node count), log(area), log(total length). Kept
                separate so size effects can be weighted/ablated rather than
                silently absorbed by a normalization choice.
Structural    — optional NetLSD heat-kernel trace, appended as extra columns
                (--structural_component netlsd).

The topological/geometric blocks operate on the *original* road graph (nodes
= intersections, edges = road segments) via osmnx, not the segment-adjacency
line graph that graph2vec_support.py / netlsd_support.py use. The attribute
block's degree<->X correlations use the line-graph degree (# of adjacent
segments), since that's the notion of "degree" MultiAttrGAT's GAT operates
on.

Usage
-----
    # Step 1: compute the raw genome (one row per city, real units)
    python stats_support.py \
        --mode        compute \
        --data_dir    ./data/raw_data \
        --cities_json ./cities.json \
        --output_csv  ./embedding_models/data/stats/output/genome_raw.csv \
        --structural_component netlsd --structural_dims 32

    # Step 2: standardize + pad/truncate to a fixed length -> plots
    python stats_support.py \
        --mode        postprocess \
        --input_csv   ./embedding_models/data/stats/output/genome_raw.csv \
        --output_csv  ./embedding_models/data/stats/output/embeddings_named.csv \
        --plot_dir    ./embedding_models/data/stats/output \
        --dimensions  128
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.expanduser("~/websites/mapedia"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NetLSD"))
# Repo root (city_learning/), so plot_selective_cross_city.py's own
# `from embedding_models.X import ...` resolves when this script is run
# directly (sys.path[0] is then embedding_models/, not the repo root).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import shapely
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from modules.city_learning.src.processing import (
    build_line_graph_edge_index,
    nlanes_to_class,
    oneway_to_class,
)
from modules.city_learning.plot_selective_cross_city import sample_by_city
from graph2vec_support import plot_similarity_matrix, plot_dendrogram, plot_scatter

N_ROAD_CLASSES  = 8     # fixed-width road_type bucket count (overflow clips into the last bin)
N_CLUSTER_SAMPLE = 3000 # cap for avg-clustering sampling on large graphs
N_DENSITY_RINGS = 5     # concentric rings for the density-gradient feature


# ── Graph construction (original road graph: nodes = intersections) ────────

def _node_coords(edges: gpd.GeoDataFrame) -> dict:
    """
    Vectorized extraction of each edge's start/end coordinates, keyed by the
    corresponding source/target node id (first occurrence wins).
    """
    coords, index = shapely.get_coordinates(edges.geometry.values, return_index=True)
    _, first_pos = np.unique(index, return_index=True)
    last_pos = np.r_[first_pos[1:] - 1, len(index) - 1]

    node_xy = {}
    for src, tgt, sp, ep in zip(edges["source"].values, edges["target"].values,
                                 first_pos, last_pos):
        node_xy.setdefault(int(src), (float(coords[sp, 0]), float(coords[sp, 1])))
        node_xy.setdefault(int(tgt), (float(coords[ep, 0]), float(coords[ep, 1])))
    return node_xy


def build_graph(edges: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """
    osmnx-compatible MultiDiGraph of the original road network (nodes =
    intersections, edges = road segments) — the original graph, not the
    segment-adjacency line graph the other embedding_models/ scripts use.
    """
    node_xy = _node_coords(edges)

    G = nx.MultiDiGraph()
    G.graph["crs"] = "epsg:4326"
    for node_id, (lon, lat) in node_xy.items():
        G.add_node(node_id, x=lon, y=lat)

    for row in edges.itertuples(index=False):
        u, v = int(row.source), int(row.target)
        if u == v:
            continue
        length = float(row.length) if not np.isnan(row.length) else 0.0
        # osmid is required by osmnx's to_undirected() to detect duplicate/
        # reciprocal edges; fall back to the row's own idx if osm_id is missing.
        osmid = int(row.osm_id) if hasattr(row, "osm_id") and not pd.isna(row.osm_id) else int(row.idx)
        G.add_edge(u, v, length=length, osmid=osmid)

    street_counts = ox.stats.count_streets_per_node(G)
    nx.set_node_attributes(G, street_counts, name="street_count")
    return G


def _area_km2(edges: gpd.GeoDataFrame) -> float:
    edges_m = edges.to_crs(epsg=3857)
    minx, miny, maxx, maxy = edges_m.total_bounds
    return float((maxx - minx) * (maxy - miny) / 1e6)


# ── Topological block ────────────────────────────────────────────────────────

def topological_features(G: nx.MultiDiGraph, area_km2: float) -> dict:
    street_counts = np.array(list(nx.get_node_attributes(G, "street_count").values()))
    n_nodes = len(street_counts)
    feats = {}

    feats["deg1_frac"]     = float(np.mean(street_counts == 1)) if n_nodes else 0.0  # = dead-end fraction
    feats["deg3_frac"]     = float(np.mean(street_counts == 3)) if n_nodes else 0.0
    feats["deg4_frac"]     = float(np.mean(street_counts == 4)) if n_nodes else 0.0
    feats["deg5plus_frac"] = float(np.mean(street_counts >= 5)) if n_nodes else 0.0

    feats["intersection_density_km2"] = float(np.sum(street_counts >= 3) / area_km2) if area_km2 > 0 else 0.0
    feats["edge_density_km2"] = float(G.number_of_edges() / area_km2) if area_km2 > 0 else 0.0

    Gu = ox.convert.to_undirected(G)
    circuity = ox.stats.circuity_avg(Gu)
    feats["circuity_avg"] = float(circuity) if circuity is not None else 0.0

    Gs = nx.Graph(G)  # simple undirected, for connectivity stats
    if Gs.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(Gs), key=len)
        feats["largest_cc_frac"] = len(largest_cc) / Gs.number_of_nodes()

        sample_n = min(N_CLUSTER_SAMPLE, Gs.number_of_nodes())
        rng = np.random.default_rng(42)
        sample_nodes = rng.choice(np.array(Gs.nodes), size=sample_n, replace=False).tolist()
        feats["avg_clustering"] = float(nx.average_clustering(Gs, nodes=sample_nodes))
    else:
        feats["largest_cc_frac"] = 0.0
        feats["avg_clustering"] = 0.0

    return feats


# ── Geometric / morphological block ─────────────────────────────────────────

def _density_gradient(edges: gpd.GeoDataFrame, n_rings: int = N_DENSITY_RINGS) -> float:
    """Slope of log(edge count) vs. distance-ring index from the centroid —
    a compact proxy for how street density falls off from the city center."""
    edges_m = edges.to_crs(epsg=3857)
    centroids = edges_m.geometry.centroid
    cx, cy = centroids.x.mean(), centroids.y.mean()
    dist = np.hypot(centroids.x.to_numpy() - cx, centroids.y.to_numpy() - cy)

    if len(dist) < n_rings or np.allclose(dist, dist[0]):
        return 0.0

    ring = pd.qcut(dist, n_rings, labels=False, duplicates="drop")
    ring_counts = pd.Series(ring).value_counts().sort_index()
    if len(ring_counts) < 2:
        return 0.0

    x = ring_counts.index.to_numpy(dtype=np.float64)
    y = np.log1p(ring_counts.to_numpy(dtype=np.float64))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def geometric_features(G: nx.MultiDiGraph, edges: gpd.GeoDataFrame) -> dict:
    feats = {}

    G = ox.bearing.add_edge_bearings(G)
    Gu = ox.convert.to_undirected(G)
    entropy = float(ox.bearing.orientation_entropy(Gu, num_bins=36))
    feats["orientation_entropy"] = entropy
    # Simplified grid-order proxy: 1 = perfectly gridded (all one axis), 0 = uniformly random.
    feats["grid_order_index"] = float(1.0 - entropy / np.log(36))

    lengths = edges["length"].to_numpy(dtype=np.float64)
    lengths = lengths[~np.isnan(lengths)]
    if len(lengths) > 0:
        q25, q50, q75, q90 = np.quantile(lengths, [0.25, 0.5, 0.75, 0.9])
        feats["edge_length_mean"] = float(np.mean(lengths))
        feats["edge_length_std"]  = float(np.std(lengths))
        feats["edge_length_q25"]  = float(q25)
        feats["edge_length_q50"]  = float(q50)
        feats["edge_length_q75"]  = float(q75)
        feats["edge_length_q90"]  = float(q90)
    else:
        for k in ("edge_length_mean", "edge_length_std", "edge_length_q25",
                  "edge_length_q50", "edge_length_q75", "edge_length_q90"):
            feats[k] = 0.0

    feats["density_gradient"] = _density_gradient(edges)
    return feats


# ── Attribute-distribution block ────────────────────────────────────────────

def attribute_features(edges: gpd.GeoDataFrame, seg_degree: np.ndarray) -> dict:
    feats = {}

    # ── road-class mix (road_type is a small pre-encoded pipeline class id) ──
    road_type = edges["road_type"].to_numpy(dtype=np.float64)
    valid_rt  = ~np.isnan(road_type)
    bucket = np.clip(np.nan_to_num(road_type, nan=N_ROAD_CLASSES - 1), 0, N_ROAD_CLASSES - 1).astype(int)
    for i in range(N_ROAD_CLASSES):
        feats[f"roadclass_{i}_frac"] = float(np.mean(bucket == i))

    # ── speed-limit distributions ────────────────────────────────────────────
    for col in ("max_speed", "min_speed"):
        vals = pd.to_numeric(edges[col], errors="coerce").to_numpy(dtype=np.float64)
        avail = ~np.isnan(vals)
        feats[f"{col}_availability"] = float(avail.mean())
        feats[f"{col}_mean"]   = float(np.mean(vals[avail])) if avail.any() else 0.0
        feats[f"{col}_median"] = float(np.median(vals[avail])) if avail.any() else 0.0

    # ── lane-count distribution ──────────────────────────────────────────────
    nlanes_cls = np.asarray(nlanes_to_class(edges["nlanes"]))  # -1 missing, 1/2/3, 0 = >3 lanes
    valid_lanes = nlanes_cls != -1
    feats["nlanes_availability"] = float(valid_lanes.mean())
    feats["nlanes_frac_1"]      = float(np.mean(nlanes_cls == 1))
    feats["nlanes_frac_2"]      = float(np.mean(nlanes_cls == 2))
    feats["nlanes_frac_3plus"]  = float(np.mean(valid_lanes & np.isin(nlanes_cls, [3, 0])))

    # ── one-way fraction ──────────────────────────────────────────────────────
    oneway_cls = oneway_to_class(edges["oneway"]).to_numpy(dtype=np.float64)
    valid_ow = ~np.isnan(oneway_cls)
    feats["oneway_availability"] = float(valid_ow.mean())
    feats["oneway_frac"] = float(np.mean(oneway_cls[valid_ow])) if valid_ow.any() else 0.0

    # ── attribute<->topology coupling ────────────────────────────────────────
    if valid_rt.sum() > 1:
        corr, _ = spearmanr(seg_degree[valid_rt], road_type[valid_rt])
        feats["degree_roadclass_corr"] = float(corr) if not np.isnan(corr) else 0.0
    else:
        feats["degree_roadclass_corr"] = 0.0

    max_speed = pd.to_numeric(edges["max_speed"], errors="coerce").to_numpy(dtype=np.float64)
    avail_speed = ~np.isnan(max_speed)
    if avail_speed.sum() > 1:
        corr, _ = spearmanr(seg_degree[avail_speed], max_speed[avail_speed])
        feats["degree_maxspeed_corr"] = float(corr) if not np.isnan(corr) else 0.0
    else:
        feats["degree_maxspeed_corr"] = 0.0

    return feats


# ── Scale block ──────────────────────────────────────────────────────────────

def scale_features(G: nx.MultiDiGraph, edges: gpd.GeoDataFrame, area_km2: float) -> dict:
    total_length_km = float(edges["length"].fillna(0).sum() / 1000.0)
    return {
        "log_node_count":      float(np.log1p(G.number_of_nodes())),
        "log_area_km2":        float(np.log1p(area_km2)),
        "log_total_length_km": float(np.log1p(total_length_km)),
    }


# ── Optional off-the-shelf structural component ─────────────────────────────

def structural_component(edges: gpd.GeoDataFrame, dims: int) -> dict:
    import netlsd
    from netlsd_support import edges_to_networkx

    # edges_to_networkx() does its own reset_index()/rename to build an 'idx'
    # column; drop the one compute() already added so it doesn't collide.
    G = edges_to_networkx(edges.drop(columns=["idx"], errors="ignore"))
    timescales = np.logspace(-2, 2, dims)
    descriptor = np.real(netlsd.heat(G, timescales=timescales, normalization="empty")).astype(np.float64)
    return {f"netlsd_{i}": float(descriptor[i]) for i in range(dims)}


# ── Compute ──────────────────────────────────────────────────────────────────

def compute(data_dir, cities_json, output_csv, structural_component_name, structural_dims):
    with open(cities_json) as f:
        cities = list(json.load(f).keys())

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows = {}
    for city in cities:
        edges_path = os.path.join(data_dir, f"{city}_edges.parquet")
        if not os.path.exists(edges_path):
            print(f"  ⚠ Skipping {city} — edges file not found")
            continue

        print(f"Processing {city} …")
        edges = gpd.read_parquet(edges_path)
        edges = edges.reset_index().rename(columns={"index": "idx"})
        N = len(edges)

        G = build_graph(edges)
        area_km2 = _area_km2(edges)
        print(f"  {G.number_of_nodes():,} nodes | {N:,} segments | area={area_km2:.1f} km2")

        edge_index = build_line_graph_edge_index(
            edges, u_col="source", v_col="target", eid_col="idx"
        ).cpu().numpy()
        seg_degree = np.bincount(edge_index[0], minlength=N) if edge_index.size else np.zeros(N)

        feats = {}
        feats.update(topological_features(G, area_km2))
        feats.update(geometric_features(G, edges))
        feats.update(attribute_features(edges, seg_degree))
        feats.update(scale_features(G, edges, area_km2))

        if structural_component_name == "netlsd":
            feats.update(structural_component(edges, structural_dims))

        rows[city] = feats

    if not rows:
        print("[ERROR] No cities processed — check data_dir / cities_json")
        sys.exit(1)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "city"
    df.to_csv(output_csv)
    print(f"\nGenome saved → {output_csv}  ({df.shape[0]} cities x {df.shape[1]} features)")
    print(f"Feature columns: {list(df.columns)}")


# ── Postprocess ───────────────────────────────────────────────────────────────

def postprocess(input_csv, output_csv, plot_dir, dimensions,
                sample_frac=None, min_per_city=1, seed=42):
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(input_csv, index_col=0)
    all_cities = list(df.index)
    raw = np.nan_to_num(df.values.astype(np.float64), nan=0.0)

    # Standardize per feature across the FULL population read from input_csv
    # (not just the sample below) — more cities/regions give a more stable
    # mean/std estimate, and it keeps scaling consistent across different
    # --sample_frac draws. Raw units span wildly different scales (fractions
    # vs. log(area) vs. correlations), so cosine similarity / PCA / t-SNE
    # need z-scored inputs to be meaningful regardless.
    mu, sd = raw.mean(axis=0), raw.std(axis=0)
    sd[sd == 0] = 1.0
    standardized = (raw - mu) / sd

    n_feat = standardized.shape[1]
    if n_feat > dimensions:
        print(f"  Note: genome has {n_feat} features > --dimensions {dimensions}; truncating.")
        embeddings_full = standardized[:, :dimensions]
    elif n_feat < dimensions:
        pad = np.zeros((standardized.shape[0], dimensions - n_feat))
        embeddings_full = np.hstack([standardized, pad])
        print(f"  Note: genome has {n_feat} features < --dimensions {dimensions}; zero-padded "
              f"(real signal is only in the first {n_feat} dims).")
    else:
        embeddings_full = standardized

    if sample_frac is not None:
        cities = sample_by_city(all_cities, sample_frac, min_per_city, seed)
        print(f"  Sampled {len(cities)}/{len(all_cities)} regions "
              f"(frac={sample_frac}, min_per_city={min_per_city}, seed={seed})")
    else:
        cities = all_cities

    idx = [all_cities.index(c) for c in cities]
    embeddings = embeddings_full[idx]

    dim_cols = [f"x_{i}" for i in range(dimensions)]
    named = pd.DataFrame(embeddings, index=cities, columns=dim_cols)
    named.index.name = "city"
    named.to_csv(output_csv)
    print(f"Standardized/padded embeddings saved → {output_csv}")

    sim_matrix = cosine_similarity(embeddings)
    pd.DataFrame(sim_matrix, index=cities, columns=cities).to_csv(
        os.path.join(plot_dir, "similarity_matrix.csv")
    )

    print("\nGenerating plots …")
    plot_similarity_matrix(sim_matrix, cities,
                           os.path.join(plot_dir, "similarity_matrix.png"),
                           method_label="Hand-Designed Genome")
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
    parser.add_argument("--input_csv",    default="./embedding_models/data/stats/output/genome_raw.csv",
                        help="[postprocess] raw genome CSV produced by --mode compute")
    parser.add_argument("--output_csv",   default=None,
                        help="[compute] where to write the raw genome; "
                             "[postprocess] where to write the standardized/padded embedding CSV")
    parser.add_argument("--plot_dir",     default="./embedding_models/data/stats/output")
    parser.add_argument("--structural_component", choices=["none", "netlsd"], default="none",
                        help="[compute] optionally append an off-the-shelf structural descriptor block")
    parser.add_argument("--structural_dims", type=int, default=32,
                        help="[compute] dimensionality of the optional structural component")
    parser.add_argument("--dimensions",   type=int, default=128,
                        help="[postprocess] output embedding length; must match the "
                             "pgvector column width (128)")
    parser.add_argument("--sample_frac",  type=float, default=None,
                        help="[postprocess] if set, sample this fraction of each city's regions "
                             "(grouped by the '<city>_r{row}_c{col}' prefix) for the output/plots "
                             "instead of using all of them — see plot_selective_cross_city.py")
    parser.add_argument("--min_per_city", type=int, default=1,
                        help="[postprocess] minimum regions kept per city when sampling")
    parser.add_argument("--seed",         type=int, default=42,
                        help="[postprocess] random seed for sampling (reproducible draws)")
    args = parser.parse_args()

    if args.mode == "compute":
        output_csv = args.output_csv or "./embedding_models/data/stats/output/genome_raw.csv"
        compute(args.data_dir, args.cities_json, output_csv,
                args.structural_component, args.structural_dims)
    else:
        output_csv = args.output_csv or "./embedding_models/data/stats/output/embeddings_named.csv"
        postprocess(args.input_csv, output_csv, args.plot_dir, args.dimensions,
                    args.sample_frac, args.min_per_city, args.seed)


if __name__ == "__main__":
    main()
