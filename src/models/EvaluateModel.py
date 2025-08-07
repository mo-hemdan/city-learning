from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(edges_df.drop(columns=['nlanes']), edges_df['nlanes'])

model = define_model()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = [round(x) for x in y_pred]

X_test_vis = X_test.join(edges[['geometry', 'single_osmid']])
X_test_vis['pred'] = y_pred
X_test_vis = X_test_vis.join(y_test)
X_test_grb = X_test_vis.groupby('single_osmid')
def majority_voting(df):
    counts = df.pred.value_counts()
    majVotes_nlanes = counts.idxmax()
    df.pred = majVotes_nlanes
    return df
X_test_final = X_test_grb.apply(majority_voting)
# %%
y_pred = X_test_final.pred
y_test_2 = X_test_final.nlanes
mae = mean_absolute_error(y_test_2, y_pred)

r2 = r2_score(y_test_2, y_pred)
mse = mean_squared_error(y_test_2, y_pred)
rmse = np.sqrt(mse)

print(mae, ', ', r2, ', ', mse, ', ', rmse)

# %%
import osmnx as ox

place = "Manhattan, New York City, USA"
gdf = ox.geocode_to_gdf(place)
polygon = gdf.geometry.iloc[0]

# Get exact bounds of the polygon
min_long, min_lat, max_long, max_lat = polygon.bounds
print("min_lat:", min_lat)
print("min_long:", min_long)
print("max_lat:", max_lat)
print("max_long:", max_long)
# %%
