from typing import List
from .agent import Agent
from ..workflow.workflow_graph import WorkFlowGraph


class TaskPlanner(Agent):

    """
    An agent responsible for planning and decomposing high-level tasks into smaller sub-tasks.
    """

    # TODO rewrite __init__ function 

    def execute(self, task: str, history: str = None, **kwargs):

        """
        Plan and decompose high-level tasks into executable task configurations.
        """
        pass


class AgentAllocator(Agent):
    """
    An agent responsible for assigning tasks to suitable agents.
    """
    # TODO rewrite __init__ function
    def execute(self, task: str, workflow_graph: WorkFlowGraph, candidate_agents: List[Agent], **kwargs):

        pass 


class WorkFlowReviewer(Agent):
    """
    An agent responsible for evaluating and reviewing the generated workflow.
    """

    # TODO rewrite __init__ function 

    def execute(self, task: str, workflow_graph: WorkFlowGraph, history: str = None, **kwargs):
        """
        Evaluate the workflow and provide feedback on potential issues.
        """
        pass

