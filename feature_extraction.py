#%%
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.models.CityLearningModel import train_and_evaluate

ox.settings.log_console = False
ox.settings.use_cache = True

place = "Manhattan, New York City, USA"
print(f"Downloading road network for {place}...")
G = ox.graph_from_place(place, network_type='drive')

# Convert to undirected for centrality
G_undirected = G.to_undirected()

# %%
# By default, weights use edge length (in meters)
edge_betweenness = nx.edge_betweenness_centrality(
    G_undirected, 
    k = 100,
    weight="length",   # You can specify another weight or None
    normalized=True    # Normalized to 0..1
)

# %%
edges = ox.graph_to_gdfs(G_undirected, nodes=False)

print(edges.columns.tolist())

print(edges['lanes'].value_counts())

def parse_lanes(val): # TODO: Assumption, we just take the first one
    if isinstance(val, list):
        try:
            return int(val[0])
        except:
            return None
    try:
        return int(val)
    except:
        return None

edges['nlanes'] = edges['lanes'].apply(parse_lanes)
# %%

# Making the dataset I want

edges['betweenness'] = None

edges['betweenness'] = edges.index.map(edge_betweenness)

# %%

import matplotlib.pyplot as plt

plt.hist(edges['nlanes'])
plt.hist(edges['betweenness'])
plt.show()


# %%

import numpy as np

corr_matrix = np.corrcoef(edges['nlanes'], edges['betweenness'])
# %%
corr_matrix[0, 1]
# %%
df_subsect = edges[['betweenness', 'nlanes']]
# %%
df_subsect.corr()
# %%


df_subsect.shape

df_subsect.dropna(inplace=True)



mae, r2, mse, rmse = train_and_evaluate(df_subsect[['betweenness']], df_subsect['nlanes'])
# %%

def parse_highway(val):
    if isinstance(val, list):
        try: return val[0]
        except: return None
    return val

edges['highway_c'] = edges['highway'].apply(parse_highway)

# %%
highway_vals_df = pd.get_dummies(edges['highway_c'])
# %%

edges = edges.join(highway_vals_df)
# %%

# Including the highway tag itself in the process
edges_df = edges[['nlanes', 'betweenness', 'primary', 'primary_link', 'residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'unclassified']].copy()


edges_df.dropna(inplace=True)
# %%
mae, r2, mse, rmse = train_and_evaluate(edges_df[['betweenness', 'primary', 'primary_link', 'residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'unclassified']], edges_df['nlanes'])

# %%
edges_df['u_degree'] = None
edges_df['v_degree'] = None

for idx, row in edges_df.iterrows():
    u, v, key = idx
    edges_df.at[idx, "u_degree"] = G_undirected.degree(u)
    edges_df.at[idx, "v_degree"] = G_undirected.degree(v)
    
#%%
mae, r2, mse, rmse = train_and_evaluate(edges_df.drop(columns=["nlanes"]),edges_df['nlanes'])

# %%
mae, r2, mse, rmse = train_and_evaluate(edges_df[['betweenness', 'u_degree', 'v_degree']],edges_df['nlanes'])

# %%
# Out-degree and In-degree doesn't show any difference. They don't reduce error except with a very small number, so we are going to ignore them for now


# %%
# Can we think of the problem as a flow network and we are trying to get the 



# %% Adding the length and oneway component
edges_df = edges_df.join(edges[['length', 'oneway']])
# %%
mae, r2, mse, rmse = train_and_evaluate(edges_df.drop(columns=["nlanes", 'u_degree', 'v_degree']), edges_df['nlanes'])


#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

X = edges_df.drop(columns=["nlanes", 'u_degree', 'v_degree'])
y = edges_df['nlanes']  

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(mae, ', ', r2, ', ', mse, ', ', rmse)

# %%
y_pred_int = [round(x) for x in y_pred]
y_pred_int



#%%
X_test_vis = X_test.join(edges[['geometry']])
X_test_vis['pred'] = y_pred_int
X_test_vis = X_test_vis.join(y_test)
#%%

# Visualize the results
import folium
import geopandas as gpd
from folium import Choropleth, GeoJson
from folium.plugins import Fullscreen

def color_by_match(row):
    return 'green' if row['pred'] == row['nlanes'] else 'red'


# Center the map on the middle of your geometries
from shapely.ops import unary_union
merged_geom = unary_union(X_test_vis.geometry)
center = merged_geom.centroid.coords[0][::-1]
m = folium.Map(location=center, zoom_start=13)
Fullscreen().add_to(m)

for _, row in X_test_vis.iterrows():
    folium.GeoJson(
        row['geometry'].__geo_interface__,
        style_function=lambda feat, color=color_by_match(row): {
            'color': color,
            'weight': 4,
            'opacity': 0.8
        },
        tooltip=f"Pred: {row['pred']} | True: {row['nlanes']}"
    ).add_to(m)


#%%
m

# %%
result = X_test_vis.groupby('nlanes').apply(lambda df: mean_absolute_error(df['nlanes'], df['pred']))
plt.plot(result)
# %%
mean_absolute_error(X_test_vis['nlanes'], X_test_vis['pred'])

# %%

def standard_osmid(val):
    if isinstance(val, list):
        return val[0]
    return val

edges['single_osmid'] = edges.osmid.apply(standard_osmid)
# %%
X_test_vis = X_test_vis.join(edges['single_osmid'])

#%%
X_test_grb = X_test_vis.groupby('single_osmid')

# %%
def majority_voting(df):
    counts = df.pred.value_counts()
    majVotes_nlanes = counts.idxmax()
    df.pred = majVotes_nlanes
    return df
X_test_final = X_test_grb.apply(majority_voting)
# %%
mean_absolute_error(X_test_final['nlanes'], X_test_final['pred'])

# %%

def color_by_match(row):
    diff = abs(row['pred'] - row['nlanes'])
    if diff == 0:
        return 'green'
    if diff == 1:
        return 'yellow'
    if diff == 2:
        return 'orange'
    if diff == 3:
        return 'red'
    return 'black'

merged_geom = unary_union(X_test_final.geometry)
center = merged_geom.centroid.coords[0][::-1]
m = folium.Map(location=center, zoom_start=13)
Fullscreen().add_to(m)

for _, row in X_test_final.iterrows():
    folium.GeoJson(
        row['geometry'].__geo_interface__,
        style_function=lambda feat, color=color_by_match(row): {
            'color': color,
            'weight': 4,
            'opacity': 0.8
        },
        tooltip=f"Pred: {row['pred']} | True: {row['nlanes']}"
    ).add_to(m)
m

# %%

