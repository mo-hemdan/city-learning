import osmnx as ox
import networkx as nx
import pandas as pd

def get_osmnx_graph(min_lat, min_long, max_lat, max_long):
    # Define the bounding box
    north, south, east, west = max_lat, min_lat, max_long, min_long
    print(f"Downloading road network for bounding box: ({south}, {west}, {north}, {east})...")

    # Download road network for the bounding box
    G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='drive')

    return G


def simplify_osmnx_graph_to_gdf(G): # TODO: Needs some adjust as it assumes many things
    # Convert to undirected for centrality computation or other analysis
    G_undirected = G.to_undirected()

    # Convert graph edges to GeoDataFrame
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

    def standard_osmid(val):
        if isinstance(val, list):
            return val[0]
        return val

    edges['single_osmid'] = edges.osmid.apply(standard_osmid)

    return edges, G_undirected





