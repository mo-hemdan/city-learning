# %%
import geopandas as gpd
import osmnx as ox

print("Loading edges GeoDataFrame...")
edges = gpd.read_parquet("../data/raw/edges.parquet")

print("Loading undirected graph...")
G_undirected = ox.load_graphml("../data/raw/G_undirected.graphml")

print("Data loaded successfully!")

# %%
import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent))  # or grandparent depending on where you are

from src.features.FeatureExtract import extract_features_from_edges

print('Extracting Features')
X, y = extract_features_from_edges(edges, G_undirected)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

from src.models.CityLearningModel import CityLearningModel

model = CityLearningModel()

model.train(X_train, y_train)

osmid_series = X_test.join(edges['single_osmid'])['single_osmid']
y_pred = model.predict(X_test, osmid_series)

mae, r2, mse, rmse = model.evaluate(y_pred, y_test)

print(mae, ', ', r2, ', ', mse, ', ', rmse)


model.save_model('../models/city_learning.pkl')
# %%
