import sys
import numpy as np
import json
import os
sys.path.append(os.path.expanduser("~/websites/mapedia"))
from modules import DBHandler

N_SEASONS, N_DAYS_OF_WEEK, N_HOURS = 4, 7, 24
SAVE_FOLDER = './data/raw_data/'
EDGES_KEEP_COLUMNS = ["source","target","mapd_id", "pgr_id","osm_id","oneway","road_type","nlanes","width","length","geometry","max_speed","min_speed"]

with open('./cities.json', 'r') as f:
    city_bounds = json.load(f)
    
db_handler = DBHandler()
db_handler.connect_to_db()
    
for city in city_bounds:
    print(f'Downloading: {city}')
    
    edges = db_handler.get_edges_enriched_df_streaming(
        min_lat= city_bounds[city]['min_lat'],
        max_lat= city_bounds[city]['max_lat'],
        min_lon= city_bounds[city]['min_lon'],
        max_lon= city_bounds[city]['max_lon']
    )
    print('Edges Columns: ', edges.columns)
    print('Converting avg_speed column into Matrix')
    # Convert to matrix: shape (425377, 671)
    speed_matrix = np.array(
        edges['avg_speed'].tolist(),
        dtype=np.float32
    ).reshape(-1, N_SEASONS, N_DAYS_OF_WEEK, N_HOURS)
    
    print('Saving to disk')
    parquet_filename = SAVE_FOLDER + f"{city}_edges.parquet"
    edges[EDGES_KEEP_COLUMNS].to_parquet(parquet_filename)
    
    numpy_filename = SAVE_FOLDER + f"{city}_speed_matrix.npy"
    np.save(numpy_filename, speed_matrix)

print('Finished Downloading all cities')
    
    
    