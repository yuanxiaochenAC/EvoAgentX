from typing import Optional
from pydantic import Field, PositiveInt

from ..core.module import BaseModule
from ..agents.agent_manager import AgentManager
from ..agents.task_planner import TaskPlanner


class WorkFlowGenerator(BaseModule):

    agent_manager: AgentManager
    task_planner: Optional[TaskPlanner] = Field(default_factory=TaskPlanner)
    num_turns: Optional[PositiveInt] = 2 

    def generate_workflow(self, goal: str, **kwargs):
        pass

