import networkx as nx
import pandas as pd

def extract_features_from_edges(edges, G):
    # Extracting the
    # By default, weights use edge length (in meters)
    edge_betweenness = nx.edge_betweenness_centrality(
        G, 
        k = 100,
        weight="length",   # You can specify another weight or None
        normalized=True    # Normalized to 0..1
    )

    edges['betweenness'] = None
    edges['betweenness'] = edges.index.map(edge_betweenness)

    # Including the highway tag itself in the process
    edges_df = edges[['nlanes', 'betweenness', 'primary', 'primary_link', 'residential', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'unclassified']].copy()
    edges_df = edges_df.join(edges[['length', 'oneway']])

    edges_df.dropna(inplace=True)

    X = edges_df.drop(columns=["nlanes"])
    y = edges_df['nlanes']

    return X, y