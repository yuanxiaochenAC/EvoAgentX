from pydantic import Field
from typing import Optional, Any

from .module import BaseModule
from .module_utils import generate_id, get_timestamp


class Message(BaseModule):

    """
    the base class for message. 

    Args: 
        content (Any): the content of the message. 
        sender (str): the sender of the message, should be an agent_id.
        receivers (str or List[str]): the receiver(s) of the message, should be a (list of) agent_name(s).
        trigger (str): the trigger of the message, normally set as the action name.
        msg_type (str): the type of the message, such as "request", "response", "command" etc. 
        wf_goal (str): the goal of the whole workflow. 
        wf_task (str): the name of a task in the workflow, i.e., the ``name`` of a WorkFlowNode instance. 
        wf_task (str): the description of a task in the workflow, i.e., the ``task`` of a WorkFlowNode instance.
    """
    
    content: Any
    agent: Optional[str] = None
    # receivers: Optional[Union[str, List[str]]] = None
    action: Optional[str] = None
    next_action: Optional[str] = None
    msg_type: Optional[str] = None
    wf_goal: Optional[str] = None
    wf_task: Optional[str] = None
    wf_task_desc: Optional[str] = None
    message_id: Optional[str] = Field(default=generate_id)
    timestamp: Optional[str] = Field(default=get_timestamp)
    
    def __str__(self) -> str:
        return self.to_str()
    
    def __eq__(self, other: "Message"):
        return self.message_id == other.message_id

    def __hash__(self):
        return self.message_id
    
    def to_str(self) -> str:
        pass 
    

