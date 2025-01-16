import threading
from enum import Enum
from typing import Union, Dict, List

from .agent import Agent
from ..core.module import BaseModule
from ..core.decorators import atomic_method
from ..storages.base import StorageHandler
from ..workflow.workflow_graph import WorkFlowGraph


class AgentState(str, Enum):
    AVAILABLE = "available"
    RUNNING = "running"


class AgentManager(BaseModule):
    """
    Responsible for creating and managing all Agent objects required for workflow operation.

    Attributes:
        storage_handler (StorageHandler): Used to load and save agents from/to storage.
        agents (List[Agent]): A list to keep track of all managed Agent instances.
        agent_states (Dict[str, AgentState]): A dictionary to track the state of each Agent by name.
    """
    agents: List[Agent] = []
    agent_states: Dict[str, AgentState] = {} # agent_name to AgentState mapping
    storage_handler: StorageHandler = None # used to load and save agent from storage.

    def init_module(self):
        self._lock = threading.Lock()
        if self.agents:
            self.check_agents()
            for agent in self.agents:
                self.agent_states[agent.name] = self.agent_states.get(agent.name, AgentState.AVAILABLE)
    
    def check_agents(self):
        # check that the names of self.agents should be unique
        duplicate_agent_names = self.check_duplicate_agents(self.agents)
        if duplicate_agent_names:
            raise ValueError(f"The agents should be unique. Found duplicate agent names: {duplicate_agent_names}!")
    
    def check_duplicate_agents(self, agents: List[Agent]) -> List[str]:
        # return the names of duplicate agents based on agent.name 
        unique_agent_names = set()
        duplicate_agent_names = set()
        for agent in agents:
            agent_name = agent.name
            if agent_name in unique_agent_names:
                duplicate_agent_names.add(agent_name)
            unique_agent_names.add(agent_name)
        return list(duplicate_agent_names)

    def list_agents(self) -> List[str]:
        """
        return all the agent names in self.agents. 
        """
        return [agent.name for agent in self.agents]
    
    def load_agent(self, agent_name: str, **kwargs) -> Agent:

        """
        load an agent from local storage through self.storage_handler and add it to self.agents. 

        Args:
            agent (str): the name of the agent.
        
        Notes: 
            - We could first load the data of an agent through self.storage_handler. 
            - Create an Agent instance through Agent.from_dict(agent_data). 
                If there is a `class_name` key in the agent_data, Agent.from_dict will create 
                an agent instance based on that class. Otherwise, the Agent class will be used.
            - Add the Agent instance to self.agents. 
        """
        if not self.storage_handler:
            raise ValueError("muse provide ``self.storage_handler`` to use ``load_agent``")
        agent_data = self.storage_handler.load_agent(agent_name=agent_name)
        agent: Agent = self.create_customize_agent(agent_data=agent_data)
        return agent

    @atomic_method
    def load_all_agents(self, **kwargs):
        """
        load all agents from storage and add them to self.agents. 
        """
        pass 
    
    def create_customize_agent(self, agent_data: dict, **kwargs) -> Agent:
        """
        create a customized agent from the provided `agent_data`. 

        Args:
            agent_data (dict): the data used to create an Agent instance, must contain the `name` and `description` keys.
        
        Returns:
            Agent: the instantiated agent instance.
        
        Notes: 
            - use CustomizeAgent.from_dict() to create the agent instance.
        """
        pass 
    
    @atomic_method
    def add_agent(self, agent: Union[str, dict, Agent], **kwargs):
        """
        add a single agent, ignore if the agent already exists (judged by the name of an agent).

        Args:
            agent (Union[str, dict, Agent]): The agent to be added.
                - Determine whether this agent should be added:
                    - If the agent's name is different from existing agents, add the agent. 
                    - If the agent's name already exists:
                        - check whether the data of this agent (if provided) is the same with existing one. If they are the same, ignore 
                - If a string is provided, it is treated as the agent's name. The agent will be loaded from storage using self.storage_handler.
                - If a dictionary is provided, CustomizeAgent.from_dict() will be used to create an Agent instance.
                - If an Agent instance is provided, it will be directly added to self.agents.
        """
        if isinstance(agent, str):
            agent_instance = self.load_agent(agent_name=agent)
        elif isinstance(agent, dict):
            agent_instance = self.create_customize_agent(agent_data=agent)
        elif isinstance(agent, Agent):
            agent_instance = agent 
        else:
            raise ValueError(f"{type(agent)} is not a supported input type of ``add_agent``. Supported types: [str, dict, Agent].")
        

    def add_agents(self, agents: List[Union[str, dict, Agent]], **kwargs):
        """
        add several agents by using self.add_agent().
        """
        pass
    
    @atomic_method
    def initialize_agents_from_workflow(self, workflow_graph: WorkFlowGraph, **kwargs):
        """
        Initialize agents from the nodes of a given WorkFlowGraph and add these agents to self.agents. 

        Args:
            workflow_graph (WorkFlowGraph): The workflow graph containing nodes with agents information.
        
        Notes:
            - The agent information is in workflow_graph.nodes: List[WorkFlowNode].
        """
        pass

    def get_agent(self, agent_name: str, **kwargs) -> Agent:
        """
        Retrieve an agent by its name from self.agents. 
        """
        pass
    
    @atomic_method
    def remove_agent(self, agent_name: str, remove_from_storage: bool=False, **kwargs):
        """
        remove an agent from self.agents (and storage). 

        Args:
            agent_name (str): the name of the agent to be removed. 
            remove_from_storage (boo): if True, remove the agent from storage if the agent is already in the storage.
        """
        pass

    def get_agent_state(self, agent_name: str) -> AgentState:
        """
        Get the state of a specific agent by its name.

        Args:
            agent_name (str): The name of the agent.

        Returns:
            AgentState: The current state of the agent, or None if not found.
        """
        pass 
    
    @atomic_method
    def set_agent_state(self, agent_name: str, new_state: AgentState) -> bool:
        """
        Update the state of a specific agent by its name.

        Args:
            agent_name (str): The name of the agent.
            new_state (AgentState): The new state to set.
        
        Returns:
            bool: True if the state was updated successfully, False otherwise.
        """
        pass

    def get_all_agent_states(self) -> Dict[str, AgentState]:
        """
        Get the states of all managed agents.

        Returns:
            Dict[str, AgentState]: A dictionary mapping agent names to their states.
        """
        pass
    
    @atomic_method
    def save_all_agents(self, **kwargs):
        """
        Save all agents to storage.
        """
        pass 
    
    @atomic_method
    def clear_agents(self):
        """
        Remove all agents from the manager.
        """
        pass 

