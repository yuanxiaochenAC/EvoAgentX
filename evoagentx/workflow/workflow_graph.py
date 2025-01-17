import threading
from enum import Enum
import networkx as nx 
from copy import deepcopy
from networkx import MultiDiGraph
from pydantic import Field, field_validator
from typing import Union, Optional, Tuple, List

from ..core.logging import logger
from ..core.module import BaseModule
from ..core.base_config import Parameter
from ..core.decorators import atomic_method


class WorkFlowNodeState(str, Enum):
    PENDING="pending"
    RUNNING="running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkFlowNode(BaseModule):

    name: str # A short name of the task. Should be unique in a single workflow
    description: str # A detailed description of the task
    inputs: List[Parameter] # inputs for the task
    outputs: List[Parameter] # outputs of the task
    agents: Optional[List[Union[str, dict]]] = None
    status: Optional[WorkFlowNodeState] = WorkFlowNodeState.PENDING

    @field_validator('agents')
    @classmethod
    def check_agent_format(cls, agents: List[Union[str, dict]]):
        for agent in agents:
            if isinstance(agent, dict):
                assert "name" in agent and "description" in agent, \
                    "must provide the name and description of an agent when specifying an agent with a dict."
        return agents

    def get_agents(self) -> List[str]:
        """
        return the agent names specified in the self.agents. 
        """
        agent_names = [] 
        for agent in self.agents:
            if isinstance(agent, str):
                agent_names.append(agent)
            elif isinstance(agent, dict):
                agent_names.append(agent["name"])
            else:
                raise TypeError(f"{type(agent)} is an unknown agent type!")
        return agent_names


class WorkFlowEdge(BaseModule):

    source: str 
    target: str 
    priority: int = 0 

    def __init__(self, edge_tuple: Optional[tuple]=(), **kwargs):
        """
        Initialize a WorkFlowEdge instance with either a tuple or keyword arguments.

        Parameters:
        ----------
            edge_tuple (tuple): a tuple containing the edge attributes in the format: (source, target, priority[optional]). 
                - source (str): the source of the edge. 
                - target (str): the target of the edge. 
                - priority (int, optional): The priority of the edge. Defaults to 0 if not provided.
            
            kwargs (dict): Key-value pairs specifying the edge attributes. These values will override those provided in `args` if both are supplied.

        Notes:
        ----------
            - Attributes provided via `kwargs` take precedence over those from the `args` tuple.
            - If `args` is empty or not provided, only `kwargs` will be used to initialize the instance.
        """
        data = self.init_from_tuple(edge_tuple)
        data.update(kwargs)
        super().__init__(**data)
    
    def init_from_tuple(self, edge_tuple: tuple) -> dict:
        if not edge_tuple:
            return {}
        keys = ["source", "target", "priority"]
        data = {k: v for k, v in zip(keys, edge_tuple)}
        return data
    
    def compare_attrs(self):
        return (self.source, self.target, self.priority)
    
    def __eq__(self, other: "WorkFlowEdge"):

        if not isinstance(other, WorkFlowEdge):
            return NotImplemented
        self_compare_attrs = self.compare_attrs()
        other_compare_attrs = other.compare_attrs()
        return all(self_attr==other_attr for self_attr, other_attr in zip(self_compare_attrs, other_compare_attrs))

    def __hash__(self):
        return hash(self.compare_attrs())
    

class WorkFlowGraph(BaseModule):

    goal: str
    nodes: Optional[List[WorkFlowNode]] = []
    edges: Optional[List[WorkFlowEdge]] = []
    graph: Optional[Union[MultiDiGraph, "WorkFlowGraph"]] = Field(default=None, exclude=True)

    def init_module(self):

        self._lock = threading.Lock()
        if not self.graph:
            self._init_from_nodes_and_edges(self.nodes, self.edges)
        elif isinstance(self.graph, MultiDiGraph):
            self._init_from_multidigraph(self.graph, self.nodes, self.edges)
        elif isinstance(self.graph, WorkFlowGraph):
            self._init_from_workflowgraph(self.graph, self.nodes, self.edges)
        else:
            raise TypeError(f"{type(self.graph)} is an unknown type for graph. Supported types: [MultiDiGraph, WorkFlowGraph]")
        self._validate_workflow_structure()

    def _init_from_nodes_and_edges(self, nodes: List[WorkFlowNode] = [], edges: List[WorkFlowEdge] = []):

        """
        Initialize the WorkFlowGraph from a set of nodes and edges. 
        """
        
        if edges and not nodes:
            raise ValueError("edges cannot be passed without nodes or a graph")
        
        self.nodes = []
        self.edges = []
        self.graph = MultiDiGraph()
        self.add_nodes(*nodes)
        self.add_edges(*edges)

    def _init_from_multidigraph(self, graph: MultiDiGraph, nodes: List[WorkFlowNode] = [], edges: List[WorkFlowEdge] = []):
        
        graph_nodes = [deepcopy(node_attrs["ref"]) for _, node_attrs in graph.nodes(data=True)]
        graph_edges = [deepcopy(edge_attrs["ref"]) for *_, edge_attrs in graph.edges(data=True)]
        graph_nodes = self.merge_nodes(graph_nodes, nodes)
        graph_edges = self.merge_edges(graph_edges, edges)
        self._init_from_nodes_and_edges(nodes=graph_nodes, edges=graph_edges)

    def _init_from_workflowgraph(self, graph: "WorkFlowGraph", nodes: List[WorkFlowNode] = [], edges: List[WorkFlowEdge] = []):
        
        graph_nodes = deepcopy(graph.nodes)
        graph_edges = deepcopy(graph.edges)
        graph_nodes = self.merge_nodes(graph_nodes, nodes)
        graph_edges = self.merge_edges(graph_edges, edges)
        self._init_from_nodes_and_edges(nodes=graph_nodes, edges=graph_edges)
    
    def _validate_workflow_structure(self):

        isolated_nodes = list(nx.isolates(self.graph))
        if len(self.graph.nodes) > 1 and isolated_nodes:
            logger.warning(f"The workflow contains isolated nodes: {isolated_nodes}")
        
        initial_nodes = self.find_initial_nodes()
        if len(self.graph.nodes) > 1 and not initial_nodes:
            error_message = "There are no initial nodes in the workflow!"
            logger.error(error_message)
            raise ValueError(error_message)

        end_nodes = self.find_end_nodes()
        if len(self.graph.nodes) > 1 and not end_nodes:
            logger.warning("There are no end nodes in the workflow")
    
    def find_initial_nodes(self):
        initial_nodes = [node for node, in_degree in self.graph.in_degree() if in_degree==0]
        return initial_nodes
    
    def find_end_nodes(self):
        end_nodes = [node for node, out_degree in self.graph.out_degree() if out_degree==0]
        return end_nodes
    
    @atomic_method
    def add_node(self, node: WorkFlowNode, **kwargs):

        if not isinstance(node, WorkFlowNode):
            raise ValueError(f"{node} is not a valid WorkFlowNode instance!")
        if self.node_exists(node.name):
            raise ValueError(f"Duplicate node names are not allowed! Found duplicate node name: {node.name}")

        self.nodes.append(node)
        self.graph.add_node(node.name, ref=node)

    @atomic_method
    def add_edge(self, edge: WorkFlowEdge, **kwargs):

        if not isinstance(edge, WorkFlowEdge):
            raise ValueError(f"{edge} is not a valid WorkFlowEdge instance!")
        for attr, node_name in zip(["source", "target"], [edge.source, edge.target]):
            if not self.node_exists(node_name):
                raise ValueError(f"{attr} node {node_name} does not exists!")
        if self.edge_exists(edge):
            raise ValueError(f"Duplicate edges are not allowed! Found duplicate edges: {edge}")
        
        self.edges.append(edge)
        self.graph.add_edge(edge.source, edge.target, ref=edge)

    def add_nodes(self, *nodes: WorkFlowNode, **kwargs):

        nodes: list = list(nodes)
        nodes.extend([kwargs.pop(var) for var in ["node", "nodes"] if var in kwargs])

        for node in nodes:
            if isinstance(node, (tuple, list)):
                for n in node:
                    self.add_node(n, **kwargs)
            else:
                self.add_node(node, **kwargs)

    def add_edges(self, *edges: WorkFlowEdge, **kwargs):

        edges: list = list(edges)
        edges.extend([kwargs.pop(var) for var in ["edge", "edges"] if var in kwargs])

        for edge in edges:
            if isinstance(edge, (tuple, list)):
                for e in edge:
                    self.add_edge(e, **kwargs)
            else:
                self.add_edge(edge, **kwargs)

    def node_exists(self, node: Union[str, WorkFlowNode]) -> bool:
        if isinstance(node, str):
            return node in self.graph.nodes
        elif isinstance(node, WorkFlowNode):
            return node.name in self.graph.nodes
        else:
            raise TypeError("node must be a str or WorkFlowNode instance")
    
    def _edge_exists(self, source: str, target: str, **attr_filters) -> bool:

        if not self.graph.has_edge(source, target):
            return False
        if attr_filters:
            for key, value in attr_filters.items():
                if key not in self.graph[source][target] or self.graph[source][target][key] != value:
                    return False
        return True
    
    def edge_exists(self, edge: Union[Tuple[str, str], WorkFlowEdge], **attr_filters) -> bool:

        """
        Check whether an edge exists in the workflow graph. The input `edge` can either be a tuple or a WorkFlowEdge instance.

        1. If a tuple is passed, it should be (source, target). The function will only determin whether there is an edge between the source node and the target node. 
        If attr_filters is passed, they will also be used to match the edge attributes. 
        2. If a WorkFlowEdge is passed, it will use the __eq__ method in WorkFlowEdge to determine 

        Parameters:
        ----------
            edge (Union[Tuple[str, str], WorkFlowEdge]):
                - If a tuple is provided, it should be in the format `(source, target)`. 
                The method will check whether there is an edge between the source and target nodes.
                If `attr_filters` are provided, they will be used to match edge attributes.
                - If a WorkFlowEdge instance is provided, the method will use the `__eq__` method in WorkFlowEdge 
                to determine whether the edge exists.

            attr_filters (dict, optional):
                Additional attributes to filter edges when `edge` is a tuple.

        Returns:
        -------
            bool: True if the edge exists and matches the filters (if provided); False otherwise.
        """
        if isinstance(edge, tuple):
            assert len(edge) == 2, "edge must be a tuple (source, target) or WorkFlowEdge instance"
            source, target = edge 
            return self._edge_exists(source, target, **attr_filters)
        elif isinstance(edge, WorkFlowEdge):
            return edge in self.edges 
        else:
            raise TypeError("edge must be a tuple (source, target) or WorkFlowEdge instance")
    
    def merge_nodes(self, nodes: List[WorkFlowNode], new_nodes: List[WorkFlowNode]):

        node_names = {node.name for node in nodes}
        for node in new_nodes:
            if node.name in node_names:
                continue
            nodes.append(node)
        return nodes
    
    def merge_edges(self, edges: List[WorkFlowEdge], new_edges: List[WorkFlowEdge]):

        for edge in new_edges:
            if edge in edges:
                continue
            edges.append(edge)
        return edges 

    def set_node_state(self, node: Union[str, WorkFlowNode], new_state: WorkFlowNodeState) -> bool:
        """
        Update the state of a specific node. 

        Args:
            node (Union[str, WorkFlowNode]): The name of a node or the node instance.
            new_state (WorkFlowNodeState): The new state to set.
        
        Returns:
            bool: True if the state was updated successfully, False otherwise.
        """
        pass 
