import networkx as nx

clustering = nx.clustering(G)  # G is your undirected graph
G[u][v]['clustering_u'] = clustering[u]
G[u][v]['clustering_v'] = clustering[v]

G_line = nx.line_graph(G)
pagerank = nx.pagerank(G_line)
G[u][v]['pagerank_u'] = pagerank[u]
G[u][v]['pagerank_v'] = pagerank[v]


def get_neighbor_highway_types(G, u):
    return [G[u][nbr].get('highway', None) for nbr in G.neighbors(u)]

# For each edge:
for u, v in G.edges():
    G[u][v]['neighbor_types_u'] = get_neighbor_highway_types(G, u)
    G[u][v]['neighbor_types_v'] = get_neighbor_highway_types(G, v)


from shapely.geometry import LineString
from geopandas import GeoSeries
import geopandas as gpd

edges_gdf = ox.utils_graph.graph_to_gdfs(G, nodes=False)
edges_gdf['geometry'] = edges_gdf['geometry'].apply(lambda g: g if isinstance(g, LineString) else g[0])

# Create 500m buffer around each road segment
edges_gdf['buffer'] = edges_gdf.geometry.buffer(0.0005)  # ~500m at equator
edges_gdf['road_density'] = edges_gdf['buffer'].apply(lambda b: edges_gdf[edges_gdf.intersects(b)].shape[0])


landuse_gdf = gpd.read_file('landuse.geojson')  # or shapefile
joined = gpd.sjoin(edges_gdf.set_geometry('geometry'), landuse_gdf, how='left', predicate='intersects')
edges_gdf['landuse_type'] = joined['landuse']  # or whatever the tag column is


poi_gdf = gpd.read_file("pois.geojson")

# Buffer around each edge
edges_gdf['buffer'] = edges_gdf.geometry.buffer(0.0005)
edges_gdf['poi_count'] = edges_gdf['buffer'].apply(lambda b: poi_gdf[poi_gdf.intersects(b)].shape[0])
