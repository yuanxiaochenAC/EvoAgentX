from typing import Tuple, Optional, List
from .agent import Agent
from ..core.message import Message, MessageType
from ..actions.task_planning import TaskPlanning, TaskPlanningOutput
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
        assert msgs is not None or action_input_data is not None, "must provide either `msgs` or `action_input_data` for TaskPlanner Agent."
        action = self.get_action(action_name=action_name)

        # update short-term memory
        if msgs is not None:
            self.short_term_memory.add_messages(msgs)
        
        # obtain action input data from short term memory
        action_input_data = action_input_data or self.get_action_inputs(action=action)
        # execute action
        execution_results: Tuple[TaskPlanningOutput, str] = action.execute(
            llm=self.llm, 
            inputs=action_input_data, 
            sys_msg=self.system_prompt,
            return_prompt=True
        )
        action_output, prompt = execution_results

        # formulate a message
        message = Message(
            content=action_output.to_str(),
            agent=self.name,
            action=action_name,
            prompt=prompt, 
            msg_type=MessageType.RESPONSE,
            wf_goal = kwargs.get("wf_goal", None),
            wf_task = kwargs.get("wf_task", None),
            wf_task_desc = kwargs.get("wf_task_desc", None)
        )

        # update short-term memory
        self.short_term_memory.add_message(message)

        return message
    
