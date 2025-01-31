from typing import Optional
from pydantic import PositiveInt

from ..core.module import BaseModule
from ..agents.agent_manager import AgentManager
from ..agents.task_planner import TaskPlanner
from ..agents.workflow_reviewer import WorkFlowReviewer


class WorkFlowGenerator(BaseModule):

    task_planner: TaskPlanner # decompose the high-level task into subtasks
    agent_manager: AgentManager # assign or generate agent(s) for a subtask
    workflow_reviewer: WorkFlowReviewer # provide reflections on the generated workflow 
    num_turns: Optional[PositiveInt] = 2 

    def generate_workflow(self, goal: str, **kwargs):
        pass

