from pydantic import Field

from ..core.module import BaseModule
from ..agents.agent_manager import AgentManager
from ..agents.agent_generator import AgentGenerator
from ..agents.workflow_agents import TaskPlannerAgent, AgentAllocatorAgent, WorkFlowReviewAgent


class WorkFlowGenerator(BaseModule):

    agent_manager: AgentManager = None
    task_planner: TaskPlannerAgent = Field(default_factory=TaskPlannerAgent)
    agent_allocator: AgentAllocatorAgent = Field(default_factory=AgentAllocatorAgent)
    agent_generator: AgentGenerator = Field(default_factory=AgentGenerator)
    workflow_reviewer: WorkFlowReviewAgent = Field(default_factory=WorkFlowReviewAgent)

    def generate_workflow(self, goal: str, **kwargs):
        pass 

