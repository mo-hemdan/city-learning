import sys
import numpy as np
import json
import os
from tqdm import tqdm
sys.path.append(os.path.expanduser("~/websites/mapedia"))
from modules import DBHandler
from modules import INTRA_CITY_LEARNING_SOURCE, INTER_CITY_LEARNING_SOURCE

N_SEASONS, N_DAYS_OF_WEEK, N_HOURS = 4, 7, 24
LEARNED_SOURCES = {INTRA_CITY_LEARNING_SOURCE, INTER_CITY_LEARNING_SOURCE}
SAVE_FOLDER = './data/raw_data/'
EDGES_KEEP_COLUMNS = ["source","target","mapd_id", "pgr_id","osm_id","oneway","road_type","nlanes","width","length","geometry","max_speed","min_speed"]
SCALAR_ATTR_SOURCE_COLUMNS = {
    "oneway": "oneway_source",
    "road_type": "road_type_source",
    "width": "width_source",
    "nlanes": "nlanes_source",
    "max_speed": "max_speed_source",
    "min_speed": "min_speed_source",
}

# with open('./cities.json', 'r') as f:
with open('./city_grids.json', 'r') as f:
    city_bounds = json.load(f)
    
db_handler = DBHandler()
db_handler.connect_to_db()

discarded = []

for city in tqdm(city_bounds, total=len(city_bounds), desc='Downloading Cities'):
    print(f'Downloading: {city}')

    edges = db_handler.get_edges_enriched_df_streaming(
        min_lat= city_bounds[city]['min_lat'],
        max_lat= city_bounds[city]['max_lat'],
        min_lon= city_bounds[city]['min_lon'],
        max_lon= city_bounds[city]['max_lon']
    )

    if edges.empty:
        print(f'  no edges found for {city}, discarding')
        discarded.append(city)
        continue

    print('Edges Columns: ', edges.columns)
    print('Edges Size: ', edges.shape)

    for attr, source_col in SCALAR_ATTR_SOURCE_COLUMNS.items():
        learned = edges[source_col].isin(LEARNED_SOURCES)
        if learned.any():
            print(f'  masking {learned.sum()} inter/intra-city learned {attr} values')
            edges.loc[learned, attr] = np.nan

    print('Converting avg_speed column into Matrix')
    # Convert to matrix: shape (425377, 671)
    speed_matrix = np.array(
        edges['avg_speed'].tolist(),
        dtype=np.float32
    ).reshape(-1, N_SEASONS, N_DAYS_OF_WEEK, N_HOURS)

    source_matrix = np.array(
        edges['avg_speed_source'].tolist(),
        dtype=np.float32
    ).reshape(-1, N_SEASONS, N_DAYS_OF_WEEK, N_HOURS)

    learned_mask = np.isin(source_matrix, list(LEARNED_SOURCES))
    print(f'  masking {learned_mask.sum()} inter/intra-city learned values out of {learned_mask.size}')
    speed_matrix[learned_mask] = np.nan

    print('edges: ', edges['max_speed'])
    print('speed_matrix', speed_matrix[:100][:100])
    
    print('Saving to disk')
    parquet_filename = SAVE_FOLDER + f"{city}_edges.parquet"
    edges[EDGES_KEEP_COLUMNS].to_parquet(parquet_filename)
    
    numpy_filename = SAVE_FOLDER + f"{city}_speed_matrix.npy"
    np.save(numpy_filename, speed_matrix)

if discarded:
    print(f'Discarding {len(discarded)} empty areas from city_grids.json: {discarded}')
    for city in discarded:
        del city_bounds[city]
    with open('./city_grids.json', 'w') as f:
        json.dump(city_bounds, f, indent=2)

print('Finished Downloading all cities')
    
    
    