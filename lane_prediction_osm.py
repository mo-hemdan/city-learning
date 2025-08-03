


#%%
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ox.settings.log_console = False
ox.settings.use_cache = True


#%%
# -----------------------------
# 1. Load Graph from OSM
# -----------------------------
place = "Manhattan, New York City, USA"
print(f"Downloading road network for {place}...")
G = ox.graph_from_place(place, network_type='drive')

# Convert to undirected for centrality
G_undirected = G.to_undirected()


# %%
# -----------------------------
# 2. Compute Node-Based Features
# -----------------------------
# print("Computing node centrality features...")
# in_degree = dict(G.in_degree())
# out_degree = dict(G.out_degree())
# betweenness = nx.betweenness_centrality(G_undirected, weight='length', normalized=True)
# closeness = nx.closeness_centrality(G_undirected)
# eigenvector = nx.eigenvector_centrality(G_undirected, max_iter=1000)


# By default, weights use edge length (in meters)
edge_betweenness = nx.edge_betweenness_centrality(
    G_undirected, 
    k = 1000,
    weight="length",   # You can specify another weight or None
    normalized=True    # Normalized to 0..1
)

# We can also use 
# 4. Parallel Edge Betweenness (if needed)
# NetworkX is single-threaded. For multi-core computation, use:

# graph-tool (very fast but uses C++ backend)

# NetworKit
#%%
# iGraph with Python bindings
import igraph as ig
import osmnx as ox

# G_nx = ox.graph_from_place("Minneapolis, Minnesota, USA", network_type="drive")
# G_nx = ox.utils_graph.get_undirected(G_nx)
G_nx = G_undirected

# Convert NetworkX to iGraph
edges = list(G_nx.edges())
nodes = list(G_nx.nodes())
node_map = {node: idx for idx, node in enumerate(nodes)}
edges_igraph = [(node_map[u], node_map[v]) for u, v in edges]

G_ig = ig.Graph(edges=edges_igraph, directed=False)
betweenness = G_ig.edge_betweenness()


# %%
# -----------------------------
# 3. Build Edge Features
# -----------------------------
print("Building edge feature dataframe...")

edge_features = []

for u, v, k, data in tqdm(G.edges(keys=True, data=True), desc="Processing edges"):
    u_feats = {
        'in_degree_u': in_degree.get(u, 0),
        'out_degree_u': out_degree.get(u, 0),
        'betweenness_u': betweenness.get(u, 0),
        'closeness_u': closeness.get(u, 0),
        'eigenvector_u': eigenvector.get(u, 0)
    }
    v_feats = {
        'in_degree_v': in_degree.get(v, 0),
        'out_degree_v': out_degree.get(v, 0),
        'betweenness_v': betweenness.get(v, 0),
        'closeness_v': closeness.get(v, 0),
        'eigenvector_v': eigenvector.get(v, 0)
    }

    combined = {
        **u_feats,
        **v_feats,
        'road_type': data.get('highway'),
        'length': data.get('length'),
        'lanes': data.get('lanes')  # target variable
    }

    edge_features.append(combined)

df = pd.DataFrame(edge_features)

# %%
# -----------------------------
# 4. Data Cleaning
# -----------------------------
df = df.dropna(subset=['lanes'])  # drop rows with no ground truth

# Convert lanes to int (some are '2;3')
def parse_lanes(val):
    try:
        if isinstance(val, list):
            val = val[0]
        if isinstance(val, str) and ';' in val:
            val = val.split(';')[0]
        return int(val)
    except:
        return np.nan

df['lanes'] = df['lanes'].apply(parse_lanes)
df = df.dropna(subset=['lanes'])

# Encode categorical road types
df = pd.get_dummies(df, columns=['road_type'])

# %%
# -----------------------------
# 5. Train ML Model
# -----------------------------
print("Training lane prediction model...")
X = df.drop(columns=['lanes'])
y = df['lanes']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
# -----------------------------
# 6. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print("\n=== Evaluation ===")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Optional: feature importances
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop Features:")
print(feat_imp.sort_values(ascending=False).head(10))
