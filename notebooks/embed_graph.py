# %%
from karateclub import (
    Graph2Vec, GL2Vec, IGE, FeatherGraph,
    GeoScattering, NetLSD, SF, FGSD, LDP, WaveletCharacteristic
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

city_names = ["jakarta", "singapore", "chicago"]

# ── Graph preparation ──────────────────────────────────────────────────────────

def prepare_graph_numerical(G, feature_cols):
    """For IGE, FeatherGraph, GeoScattering, WaveletCharacteristic"""
    G_new = G.copy()
    for node, data in G_new.nodes(data=True):
        scalar_feats = [float(data.get(f, 0)) for f in feature_cols]
        if 'speed' in data:
            full_vec = scalar_feats + data['speed'].tolist()  # (8 + 672,)
        else:
            full_vec = scalar_feats
        G_new.nodes[node]["feature"] = full_vec
    return G_new

# ── Model runner ───────────────────────────────────────────────────────────────

def run_model(name, model, graphs):
    try:
        model.fit(graphs)
        emb = model.get_embedding()
        print(f"✓ {name}: {emb.shape}")
        return emb
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        return None

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_embedding(name, embeddings, labels):
    print(f"\n{'='*40}\nModel: {name}")
    sim_df  = pd.DataFrame(cosine_similarity(embeddings),   index=labels, columns=labels)
    dist_df = pd.DataFrame(euclidean_distances(embeddings), index=labels, columns=labels)
    print("Cosine similarity:\n",  sim_df.round(3))
    print("Euclidean distance:\n", dist_df.round(3))
    print(f"Embedding std (discriminability): {embeddings.std(axis=0).mean():.4f}")

# ── Run all models ─────────────────────────────────────────────────────────────

# Structure-only
embeddings_g2v    = run_model("Graph2Vec",          Graph2Vec(wl_iterations=2, dimensions=128, epochs=100, learning_rate=0.025, attributed=False), graphs)
embeddings_gl2vec = run_model("GL2Vec",             GL2Vec(wl_iterations=2, dimensions=128, epochs=100, learning_rate=0.025), graphs)
embeddings_netlsd = run_model("NetLSD",             NetLSD(), graphs)
embeddings_sf     = run_model("SF",                 SF(), graphs)
embeddings_fgsd   = run_model("FGSD",               FGSD(), graphs)
embeddings_ldp    = run_model("LDP",                LDP(), graphs)

# Node-attribute aware (use numerical graphs)
embeddings_ige     = run_model("IGE",               IGE(dimensions=128, base_graph_order=2), graphs_numerical)
embeddings_feather = run_model("FeatherGraph",      FeatherGraph(order=5, eval_points=25, theta_max=2.5), graphs_numerical)
embeddings_geo     = run_model("GeoScattering",     GeoScattering(), graphs_numerical)
embeddings_wavelet = run_model("WaveletCharacteristic", WaveletCharacteristic(), graphs_numerical)

# ── Evaluate all ───────────────────────────────────────────────────────────────

results = {
    "Graph2Vec":           embeddings_g2v,
    "GL2Vec":              embeddings_gl2vec,
    "IGE":                 embeddings_ige,
    "FeatherGraph":        embeddings_feather,
    "GeoScattering":       embeddings_geo,
    "WaveletCharacteristic": embeddings_wavelet,
    "NetLSD":              embeddings_netlsd,
    "SF":                  embeddings_sf,
    "FGSD":                embeddings_fgsd,
    "LDP":                 embeddings_ldp,
}

for name, emb in results.items():
    if emb is not None:
        evaluate_embedding(name, emb, city_names)