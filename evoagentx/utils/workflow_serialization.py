"""
Utilities for serializing and deserializing workflow graphs.
"""
from copy import deepcopy
from networkx import MultiDiGraph
from evoagentx.workflow.workflow_graph import WorkFlowGraph

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

def workflow_graph_to_dict(workflow_graph):
    """Convert a WorkFlowGraph to a dictionary with built-in Python types.
    
    This function converts a WorkFlowGraph and all its components (nodes, edges, etc.)
    into a fully serializable dictionary containing only built-in Python types.
    
    Parameters
    ----------
    workflow_graph : WorkFlowGraph
        The workflow graph to convert
    
    Returns
    -------
    dict
        A dictionary representation of the workflow graph with all components
        converted to built-in Python types
    """
    # First use the to_dict method from BaseModule
    workflow_dict = workflow_graph.to_dict(exclude_none=True, ignore=["graph"])
    
    # If the graph attribute exists and is a MultiDiGraph, convert it to dict
    if workflow_graph.graph is not None:
        if isinstance(workflow_graph.graph, dict):
            # The graph is already a dict
            workflow_dict["graph"] = workflow_graph.graph
        else:
            # Convert MultiDiGraph to dict
            workflow_dict["graph"] = multidigraph_to_dict(workflow_graph.graph)
    
    return workflow_dict

def dict_to_workflow_graph(workflow_dict):
    """Convert a dictionary back to a WorkFlowGraph.
    
    This function reconstructs a WorkFlowGraph from a dictionary representation.
    
    Parameters
    ----------
    workflow_dict : dict
        Dictionary representation of a workflow graph
    
    Returns
    -------
    WorkFlowGraph
        The reconstructed WorkFlowGraph
    """
    # Make a copy to avoid modifying the original dict
    workflow_data = deepcopy(workflow_dict)
    
    # If 'graph' key exists in the dictionary
    graph_data = workflow_data.pop("graph", None)
    
    # Use from_dict method of BaseModule to create the WorkFlowGraph
    workflow_graph = WorkFlowGraph.from_dict(workflow_data)
    
    # If graph data exists, convert it back to MultiDiGraph and set it
    if graph_data is not None:
        workflow_graph.graph = dict_to_multidigraph(graph_data)
    
    return workflow_graph 