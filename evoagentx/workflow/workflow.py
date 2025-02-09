from pydantic import Field
from typing import Optional
from ..core.module import BaseModule
from ..core.module_utils import generate_id
# from ..models.model_configs import LLMConfig
from ..models.base_model import BaseLLM
from ..agents.agent_manager import AgentManager
from ..storages.base import StorageHandler
from .environment import Environment
from .workflow_manager import WorkFlowManager
from .workflow_graph import WorkFlowNode, WorkFlowGraph, WorkFlowNodeState


class WorkFlow(BaseModule):

    graph: WorkFlowGraph
    agent_manager: AgentManager
    llm: Optional[BaseLLM] = None
    workflow_manager: WorkFlowManager = Field(default=None, description="Responsible for task and action scheduling for workflow execution")
    environment: Environment = Field(default_factory=Environment)
    storage_handler: StorageHandler = None
    workflow_id: str = Field(default_factory=generate_id)
    version: int = 0 

    def init_module(self):
        if self.workflow_manager is None:
            if self.llm is None:
                raise ValueError("Must provide `llm` when `workflow_manager` is None")
            self.workflow_manager = WorkFlowManager(llm=self.llm)

    def execute(self, **kwargs):
        """
        Execute the workflow in a loop:
            - Check whether the workflow is completed. If the workflow is completed or there is failed task, stop execution.
            - Use self.workflow_manager.schedule_next_task to get the next task. 
            - Use self.graph.set_node_state to set the state of that task node as WorkFlowNodeState.RUNNING.
            - Use self.workflow_manager.schedule_next_action to execute (multiple) action(s) to finish this task:
                - schedule_next_action will return an NextAction object
                - Obtain the agent from self.agent_manager, set the agent state to AgentState.RUNNING, 
                    initialize a new short_term_memory if the agent is new in current ``workflow''. 
                    retrieve all relevant context from environment and execute the action.
                - Publish the action result to self.environment
                - Set the agent state to AgentState.AVAILABLE. 
            - Use self.graph.set_node_state to update the state of the task.
            - If the state of the current task is WorkFlowNodeState.FAILED, stop execution and return the error message. 
        If the workflow is successfully executed, update the agent's long_term_memory based on the short_term_memory.
        """
        failed = False
        while not self.graph.is_complete and not failed:
            task: WorkFlowNode = self.workflow_manager.schedule_next_task(graph=self.graph, env=self.environment)
            if task is None:
                break