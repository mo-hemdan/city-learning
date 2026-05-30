import geopandas as gpd
import numpy as np
import psutil, os, pickle
from utils import edges_df_to_line_graph

def get_memory_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 3

def prepare_graph_numerical(G, feature_cols):
    G_new = G.copy()
    for node, data in G_new.nodes(data=True):
        scalar_feats = [float(data.get(f, 0)) for f in feature_cols]
        if 'speed' in data:
            full_vec = scalar_feats + data['speed'].tolist()
        else:
            full_vec = scalar_feats
        G_new.nodes[node]["feature"] = full_vec
    return G_new


save_dir = "processed_graphs"
os.makedirs(save_dir, exist_ok=True)

city_names   = ['jakarta', 'singapore', 'chicago', 'NewYorkCity', 'sanFrancisco', 'washingtonDC']
feature_cols = ['oneway', 'road_type', 'nlanes', 'width', 'length', 'max_speed', 'min_speed']
keep_columns = ['osm_id', 'oneway', 'road_type', 'nlanes', 'width', 'length', 'geometry', 'max_speed', 'min_speed']

for city in city_names:
    save_path = os.path.join(save_dir, f"{city}.pkl")

    if os.path.exists(save_path):
        print(f"Skipping {city} — already saved at {save_path}")
        continue

    mem_start = get_memory_gb()
    print(f"\nProcessing {city}...  [RAM before: {mem_start:.2f} GB]")

    # 1. Load edges and speed matrix
    edges        = gpd.read_parquet(f"raw_data/{city}.parquet")
    matrices     = np.load(f"raw_data/{city}_speed_matrices.npz")
    speed_matrix = matrices['speed']
    del matrices
    print(f"  speed_matrix shape: {speed_matrix.shape}  [{speed_matrix.nbytes/1024**3:.2f} GB]  [RAM: {get_memory_gb():.2f} GB]")

    # 2. Build line graph
    L, G = edges_df_to_line_graph(edges, attribtues=keep_columns)
    del edges
    print(f"  After line graph: [RAM: {get_memory_gb():.2f} GB]")

    # 3. Attach speed vectors to line graph nodes
    edge_list   = list(G.edges())
    edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}

    missing = 0
    for node in L.nodes():
        idx = edge_to_idx.get(node) or edge_to_idx.get((node[1], node[0]))
        if idx is not None:
            L.nodes[node]['speed'] = speed_matrix[idx]
        else:
            missing += 1

    del speed_matrix, edge_to_idx, edge_list, G
    print(f"  After speed attach: [RAM: {get_memory_gb():.2f} GB]  (missing: {missing})")

    # 4. Build numeric feature vectors
    G_prepared = prepare_graph_numerical(L, feature_cols)
    del L
    print(f"  Done — {G_prepared.number_of_nodes()} nodes  [RAM: {get_memory_gb():.2f} GB]")

    # 5. Save to disk and free memory immediately
    with open(save_path, "wb") as f:
        pickle.dump(G_prepared, f)
    del G_prepared
    print(f"  Saved to {save_path}  [RAM after free: {get_memory_gb():.2f} GB]")

print("\nAll cities processed.")