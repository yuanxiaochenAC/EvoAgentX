from pydantic import Field

from ..core.parser import Parser
from ..models.base_model import BaseLLM
from ..core.module import BaseModule
from ..actions.action import Action
from ..agents.agent_manager import AgentManager
from .environment import Environment
from .workflow_graph import WorkFlowNode, WorkFlowGraph


class Scheduler(Action):
    """
    An interface for the workflow schedulers
    """
    pass


class TaskScheduler(Action):

    """
    Determines the next task to execute based on the workflow graph and node statuses.
    """
    def __init__(self, **kwargs):
        name = kwargs.pop("name", None) if "name" in kwargs else "todo_default_name"
        description = kwargs.pop("description", None) if "description" in kwargs else "todo_default_description"
        super().__init__(name=name, description=description, **kwargs)

    def execute(self, graph: WorkFlowGraph, env: Environment = None, **kwargs) -> str:
        """
        Determine the next executable tasks.

        Args:
            graph (WorkFlowGraph): The workflow graph.
        
        Returns:
            str: the name of the task to execute. 
        """
        pass 


class NextAction(Parser):
    agent: str
    action: str 


class ActionScheduler(Action):

    """
    Determines the next action(s) to execute for a given task using an LLM.
    """
    def __init__(self, **kwargs):
        name = kwargs.pop("name", None) or "todo_default_name"
        description = kwargs.pop("description", None) or "todo_default_description"
        prompt = kwargs.pop("prompt", None) or "todo_default_prompt"
        parser = NextAction
        super().__init__(name=name, description=description, prompt=prompt, parser=parser, **kwargs)
    
    def execute(self, task: str, graph: WorkFlowGraph, agent_manager: AgentManager, env: Environment = None, llm: BaseLLM = None, **kwargs) -> NextAction:
        """
        Determine the next actions to take for the given task. 
        Implement this using ReAct by default.
        If the last message stored in ``next_actions`` specifies the ``next_actions``, choose an action from these actions to execute. 

        Returns:
            NextAction: The next action to execute for the task.
        """
        pass 


class WorkFlowManager(BaseModule):
    """
    Responsible for the scheduling and decision-making when executing a workflow. 

    Attributes:
        task_scheduler (TaskScheduler): Determines the next task(s) to execute based on the workflow graph and node states.
        action_scheduler (ActionScheduler): Determines the next action(s) to take for the selected task using an LLM.
    """
    action_scheduler: ActionScheduler = Field(default_factory=ActionScheduler)
    task_scheduler: TaskScheduler = Field(default_factory=TaskScheduler)

    def schedule_next_task(self, graph: WorkFlowGraph, env: Environment = None, **kwargs) -> WorkFlowNode:
        """
        Return the next task to execute. 
        """
        if graph.is_complete:
            return None
        task_name = self.task_scheduler.execute(graph=graph, env=env)
        task: WorkFlowNode = graph.get_node(task_name)
        return task

    def schedule_next_action(self, task: str, graph: WorkFlowGraph, agent_manager: AgentManager, env: Environment = None, llm: BaseLLM = None, **kwargs) -> NextAction:
        """
        return the next action to execute. If the task is completed, return None.
        """
        pass

