#%%
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

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
# %%
X_train, X_test, y_train, y_test = train_test_split(df_subsect[['betweenness']], df_subsect['nlanes'])
# %%

model = RandomForestRegressor()
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %%

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(mae, ', ', r2, ', ', mse, ', ', rmse)

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
X_train, X_test, y_train, y_test = train_test_split(edges_df[['betweenness', 'primary', 'primary_link', 'residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'unclassified']], edges_df['nlanes'])
# %%

model = RandomForestRegressor()
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %%

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(mae, ', ', r2, ', ', mse, ', ', rmse)
# %%
