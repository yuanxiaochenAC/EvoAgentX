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

