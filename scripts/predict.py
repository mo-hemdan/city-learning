# %%
import geopandas as gpd
import osmnx as ox
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))  # or grandparent depending on where you are
from src.models.CityLearningModel import CityLearningModel
from src.features.FeatureExtract import extract_features_from_edges


print("Loading edges GeoDataFrame...")
edges = gpd.read_parquet("../data/raw/edges.parquet")

print("Loading undirected graph...")
G_undirected = ox.load_graphml("../data/raw/G_undirected.graphml")

print("Data loaded successfully!")

def predict(edges, G_undirected):
    print('Extracting Features')
    X, y = extract_features_from_edges(edges, G_undirected)

    model = CityLearningModel()

    model.load_model('../models/city_learning.pkl')

    osmid_series = X.join(edges['single_osmid'])['single_osmid']
    y_pred = model.predict(X, osmid_series)

    mae, r2, mse, rmse = model.evaluate(y_pred, y)

    print(mae, ', ', r2, ', ', mse, ', ', rmse)
    return y_pred

# %%
y_pred = predict(edges, G_undirected)