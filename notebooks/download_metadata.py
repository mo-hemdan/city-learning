# %%
import geopandas as gpd
import os
from DBHandler import DBHandler
import json 

save_dir = "raw_data"
os.makedirs(save_dir, exist_ok=True)

db = DBHandler()
# db.connect_to_db()

with open('cities.json', 'r', encoding='utf-8') as f:
    cities = json.load(f)

for city, bbox in cities.items():
    parquet_path = f"{save_dir}/{city}.parquet"

    if os.path.exists(parquet_path):
        print(f"Skipping {city} — already saved")
        continue

    print(f"\nDownloading {city}...")
    edges = db.get_edges_enriched_df_streaming(
            min_lat=bbox['min_lat'],
            max_lat=bbox['max_lat'],
            min_lon=bbox['min_lon'],
            max_lon=bbox['max_lon']
        )    
    break
    edges.to_parquet(parquet_path, index=False)
    print(f"  Saved {len(edges)} edges → {parquet_path}")
# %%
edges['']