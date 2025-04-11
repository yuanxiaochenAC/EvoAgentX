"""
MultiDiGraph to Dictionary Conversion Utilities

This module provides functions to convert NetworkX MultiDiGraph objects to Python dictionaries
and vice versa. These functions are useful for:

1. Serialization: Convert a MultiDiGraph to a dictionary that can be easily stored as JSON
2. Deserialization: Recreate a MultiDiGraph from a dictionary representation

The dictionary format preserves all graph attributes, nodes, edges, and their respective attributes.

Example usage:
-------------
    import networkx as nx
    from test_multidigraph_dict import multidigraph_to_dict, dict_to_multidigraph
    
    # Create a graph
    G = nx.MultiDiGraph(name="example")
    G.add_node(1, label="Node 1")
    G.add_edge(1, 2, key="edge1", weight=3.0)
    
    # Convert to dictionary for storage
    graph_dict = multidigraph_to_dict(G)
    
    # Store in a file, database, etc.
    import json
    with open("graph.json", "w") as f:
        json.dump(graph_dict, f)
    
    # Later, load and restore
    with open("graph.json", "r") as f:
        loaded_dict = json.load(f)
    
    # Recreate the graph
    restored_graph = dict_to_multidigraph(loaded_dict)
"""

import unittest
import networkx as nx
from copy import deepcopy
from networkx import MultiDiGraph


def multidigraph_to_dict(G):
    """Convert a NetworkX MultiDiGraph to a dictionary.
    
    This function extracts all graph metadata, nodes, and edges (with their keys)
    into a serializable dictionary format.
    
    Parameters
    ----------
    G : MultiDiGraph
        The graph to convert
    
    Returns
    -------
    dict
        A dictionary representation of the graph with the following structure:
        {
            'directed': True,  # Always true for MultiDiGraph
            'multigraph': True,  # Always true for MultiDiGraph
            'graph': dict,  # Graph attributes
            'nodes': list of dicts,  # Node attributes
            'edges': list of dicts,  # Edge attributes with source, target, key
        }
    """
    if not isinstance(G, MultiDiGraph):
        raise TypeError("Input graph must be a NetworkX MultiDiGraph")
    
    # Initialize the dictionary with graph metadata
    data = {
        'directed': True,  # MultiDiGraph is always directed
        'multigraph': True,  # MultiDiGraph is always a multigraph
        'graph': deepcopy(G.graph),  # Graph attributes
        'nodes': [],
        'edges': []
    }
    
    # Add nodes with their attributes
    for node, node_attrs in G.nodes(data=True):
        node_dict = {'id': node}
        node_dict.update(deepcopy(node_attrs))
        data['nodes'].append(node_dict)
    
    # Add edges with their attributes and keys
    for u, v, key, edge_attrs in G.edges(data=True, keys=True):
        edge_dict = {
            'source': u,
            'target': v,
            'key': key
        }
        edge_dict.update(deepcopy(edge_attrs))
        data['edges'].append(edge_dict)
    
    return data


def dict_to_multidigraph(data):
    """Convert a dictionary representation back to a NetworkX MultiDiGraph.
    
    Parameters
    ----------
    data : dict
        Dictionary containing graph data with nodes, edges, and attributes
    
    Returns
    -------
    G : MultiDiGraph
        A MultiDiGraph reconstructed from the dictionary
    """
    # Validate the input data
    if not isinstance(data, dict):
        raise TypeError("Input data must be a dictionary")
    
    # Check if the data represents a multigraph and directed graph
    is_directed = data.get('directed', True)
    is_multigraph = data.get('multigraph', True)
    
    if not (is_directed and is_multigraph):
        # If the data doesn't indicate it's a directed multigraph, warn but continue
        import warnings
        warnings.warn("Input data does not specify a directed multigraph, but forcing creation of MultiDiGraph")
    
    # Create a new MultiDiGraph
    G = MultiDiGraph()
    
    # Set graph attributes
    if 'graph' in data:
        G.graph.update(deepcopy(data['graph']))
    
    # Add nodes with attributes
    for node_data in data.get('nodes', []):
        # Copy the node data dict
        node_attrs = deepcopy(node_data)
        # Extract the ID (remove it from attributes)
        node_id = node_attrs.pop('id')
        # Add the node with its attributes
        G.add_node(node_id, **node_attrs)
    
    # Add edges with attributes and keys
    for edge_data in data.get('edges', []):
        # Copy the edge data dict
        edge_attrs = deepcopy(edge_data)
        # Extract source, target, and key
        source = edge_attrs.pop('source')
        target = edge_attrs.pop('target')
        edge_key = edge_attrs.pop('key', None)  # Key is optional
        
        # Add the edge with its key and attributes
        G.add_edge(source, target, key=edge_key, **edge_attrs)
    
    return G


class TestMultiDiGraphDict(unittest.TestCase):
    """Tests for MultiDiGraph to dictionary conversion and back."""
    
    def test_conversion(self):
        """Test converting a MultiDiGraph to a dict and back."""
        # Create a test graph
        G = MultiDiGraph(name="test_graph", day="Friday")
        
        # Add nodes with attributes
        G.add_node(1, name="Node 1", value=100)
        G.add_node(2, name="Node 2", value=200)
        G.add_node("complex_key", name="Node 3", complex=True)
        
        # Add edges with attributes and specific keys
        G.add_edge(1, 2, key="edge1", weight=3.14, label="Edge 1-2")
        G.add_edge(1, 2, key="edge2", weight=2.71, label="Another 1-2")
        G.add_edge(2, "complex_key", weight=1.5)
        G.add_edge("complex_key", 1, key=5, priority="high")
        
        # Convert to dictionary
        graph_dict = multidigraph_to_dict(G)
        
        # Convert back to MultiDiGraph
        G2 = dict_to_multidigraph(graph_dict)
        
        # Tests that conversion preserves structure
        self.assertEqual(G.graph, G2.graph)
        self.assertEqual(set(G.nodes()), set(G2.nodes()))
        
        # Check node attributes
        for node in G.nodes():
            self.assertEqual(G.nodes[node], G2.nodes[node])
        
        # Check edges (with keys)
        # We need to compare edges without sorting since keys can be of different types
        edges1 = set((u, v) for u, v, k in G.edges(keys=True))
        edges2 = set((u, v) for u, v, k in G2.edges(keys=True))
        self.assertEqual(edges1, edges2)
        
        # For each edge, check that all keys from G exist in G2 and their attributes match
        for u, v in edges1:
            keys1 = set(G[u][v].keys())
            keys2 = set(G2[u][v].keys())
            self.assertEqual(keys1, keys2)
            
            for k in keys1:
                self.assertEqual(G[u][v][k], G2[u][v][k])
    
    def test_empty_graph(self):
        """Test conversion of an empty graph."""
        G = MultiDiGraph(name="empty_graph")
        
        # Convert to dictionary and back
        graph_dict = multidigraph_to_dict(G)
        G2 = dict_to_multidigraph(graph_dict)
        
        # Verify the attributes were preserved
        self.assertEqual(G.graph, G2.graph)
        self.assertEqual(set(G.nodes()), set(G2.nodes()))
        self.assertEqual(list(G.edges()), list(G2.edges()))


if __name__ == "__main__":
    unittest.main() 