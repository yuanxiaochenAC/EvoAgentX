from typing import Optional, List
from .agent import Agent
from ..core.message import Message, MessageType
from ..actions.task_planning import TaskPlanning
from ..prompts.task_planner import TASK_PLANNER


class TaskPlanner(Agent):

    """
    An agent responsible for planning and decomposing high-level tasks into smaller sub-tasks.
    """
    def __init__(self, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else TASK_PLANNER["name"]
        description = kwargs.pop("description") if "description" in kwargs else TASK_PLANNER["description"]
        system_prompt = kwargs.pop("system_prompt") if "system_prompt" in kwargs else TASK_PLANNER["system_prompt"]
        actions = kwargs.pop("actions") if "actions" in kwargs else [TaskPlanning()]
        super().__init__(name=name, description=description, system_prompt=system_prompt, actions=actions, **kwargs)
    
    def execute(self, action_name: str, msgs: Optional[List[Message]] = None, action_input_data: Optional[dict] = None, **kwargs) -> Message:

        """
        Plan and decompose high-level tasks into executable task configurations.
        """
        message = super().execute(
            action_name=action_name, 
            action_input_data=action_input_data, 
            msgs=msgs, 
            return_msg_type=MessageType.RESPONSE, 
            **kwargs  
        )

        return message
