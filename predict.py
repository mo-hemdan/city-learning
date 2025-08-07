# %%
import osmnx as ox
from modules.city_learning.src.models.CityLearningModel import CityLearningModel
from modules.city_learning.src.features.FeatureExtract import extract_features_from_edges
from modules.city_learning.src.utils.utils import simplify_osmnx_graph_to_gdf
from shapely.geometry import box

ox.settings.log_console = False
ox.settings.use_cache = True


def infer_metadata(min_lon, min_lat, max_lon, max_lat):
    # place = "Manhattan, New York City, USA"
    # print(f"Downloading road network for {place}...")
    # G = ox.graph_from_place(place, network_type='drive')
    # Create a polygon from the bounding box

    # Download road network within the bounding box polygon
    print("Downloading road network from bounding box polygon...")
    G = ox.graph_from_polygon(box(min_lon, min_lat, max_lon, max_lat), network_type='drive')

    print('Simplifying Graph into Geodataframe')
    edges, G_undirected = simplify_osmnx_graph_to_gdf(G)
    import pandas as pd
    def parse_highway(val):
        if isinstance(val, list):
            try: return val[0]
            except: return None
        return val

    edges['highway_c'] = edges['highway'].apply(parse_highway)

    # Define all possible highway categories (even if some aren't present in the current dataset)
    ALL_HIGHWAY_CATEGORIES = [
        'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 
        'residential', 'unclassified', 'motorway', 'service', 'living_street', 'track', 'footway'
    ]

    # Convert to categorical with fixed categories
    edges['highway_c'] = pd.Categorical(edges['highway_c'], categories=ALL_HIGHWAY_CATEGORIES)

    # Now get_dummies will generate all expected columns
    highway_vals_df = pd.get_dummies(edges['highway_c'])

    edges = edges.join(highway_vals_df)
    print(edges.columns)

    # edges.drop(columns=['osmid', 'highway', 'maxspeed', 'name', 'reversed', 'ref', 'access', 'width', 'bridge', 'lanes'], inplace=True)
    
    # print(edges.columns)

    """
    edges.to_parquet("raw/edges.parquet")
    ox.save_graphml(G_undirected, filepath="raw/G_undirected.graphml")

    print("Loading edges GeoDataFrame...")
    edges = gpd.read_parquet("../data/raw/edges.parquet")

    print("Loading undirected graph...")
    G_undirected = ox.load_graphml("../data/raw/G_undirected.graphml")

    print("Data loaded successfully!")
    """

    print('Extracting Features')
    X, y = extract_features_from_edges(edges, G_undirected)

    model = CityLearningModel()

    model.load_model('./models/city_learning.pkl')

    osmid_series = X.join(edges['single_osmid'])['single_osmid']
    y_pred = model.predict(X, osmid_series)

    mae, r2, mse, rmse = model.evaluate(y_pred, y)

    print(mae, ', ', r2, ', ', mse, ', ', rmse)

    y_pred.name = 'y_pred'
    result = edges.merge(y_pred, how='right', left_index=True, right_index=True)
    return result[['y_pred', 'geometry']].reset_index(drop=True)



# y_pred = infer_metadata(106.8652, -6.2198, 106.8835, -6.2058)
# %%
