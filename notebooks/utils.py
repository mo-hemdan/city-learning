import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import psutil, os

# ── Line graph builder (recovered from your past sessions) ────────────────────
def edges_df_to_line_graph(df, attribtues=None):
    """
    Convert a GeoDataFrame of road edges into a NetworkX line graph.
    Each road segment becomes a node; two nodes connect if they share an intersection.
    
    Parameters:
        df         : GeoDataFrame with 'u', 'v' columns (OSMnx edge format)
        attribtues : list of columns to carry over as node attributes
    Returns:
        L : line graph (NetworkX)
        G : original graph (NetworkX)
    """
    # Build original directed graph
    edge_attrs = {}
    if attribtues:
        keep = [c for c in attribtues if c in df.columns and c not in ('geometry',)]
        for _, row in df.iterrows():
            edge_attrs[(row['u'], row['v'])] = {c: row[c] for c in keep}

    G = nx.DiGraph()
    for _, row in df.iterrows():
        u, v = row['u'], row['v']
        attrs = edge_attrs.get((u, v), {})
        G.add_edge(u, v, **attrs)

    # Convert to line graph — each edge becomes a node
    L = nx.line_graph(G)

    # Copy edge attributes from G onto L's nodes
    for node in L.nodes():
        u, v = node[0], node[1]
        if G.has_edge(u, v):
            for k, val in G[u][v].items():
                L.nodes[node][k] = val

    return L, G


# ── Memory helper ─────────────────────────────────────────────────────────────
def get_memory_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 3


# ── Feature builder for karateclub numerical models ───────────────────────────
def prepare_graph_numerical(G, feature_cols):
    """
    Attach a numeric feature vector to each node.
    Vector = scalar road attributes + 672-dim speed profile.
    For IGE, FeatherGraph, GeoScattering, WaveletCharacteristic.
    """
    G_new = G.copy()
    for node, data in G_new.nodes(data=True):
        scalar_feats = []
        for f in feature_cols:
            val = data.get(f, 0)
            try:
                scalar_feats.append(float(val))
            except (TypeError, ValueError):
                scalar_feats.append(0.0)

        if 'speed' in data:
            full_vec = scalar_feats + list(data['speed'])   # (7 + 672,)
        else:
            full_vec = scalar_feats + [0.0] * 672

        G_new.nodes[node]["feature"] = full_vec
    return G_new