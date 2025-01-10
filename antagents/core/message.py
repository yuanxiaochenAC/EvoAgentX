from enum import Enum
from pydantic import Field
from datetime import datetime
from typing import Optional, Callable, Any, List

from .module import BaseModule
from .module_utils import generate_id, get_timestamp

class MessageType(Enum):
    
    REQUEST = "request"
    RESPONSE = "response"
    COMMAND = "command"
    ERROR = "error"
    UNKNOWN = "unknown"


class Message(BaseModule):

    """
    the base class for message. 

    Args: 
        content (Any): the content of the message. 
        agent (str): the sender of the message, should be an agent_id.
        action (str): the trigger of the message, normally set as the action name.
        prompt (str): the prompt used to obtain the generated text. 
        next_actions (List[str]): the following actions. 
        msg_type (str): the type of the message, such as "request", "response", "command" etc. 
        wf_goal (str): the goal of the whole workflow. 
        wf_task (str): the name of a task in the workflow, i.e., the ``name`` of a WorkFlowNode instance. 
        wf_task_desc (str): the description of a task in the workflow, i.e., the ``description`` of a WorkFlowNode instance.
        message_id (str): the unique identifier of the message. 
        timestamp (str): the timestame of the message. 
    """
    
    content: Any
    agent: Optional[str] = None
    # receivers: Optional[Union[str, List[str]]] = None
    action: Optional[str] = None
    prompt: Optional[str] = None
    next_actions: Optional[List[str]] = None
    msg_type: Optional[MessageType] = MessageType.UNKNOWN
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
        # TODO 
        pass 

    @classmethod
    def sort_by_timestamp(cls, messages: List['Message'], reverse: bool = False) -> List['Message']:
        """
        sort the messages based on the timestamp. 

        Args: 
            messages (List[Message]): the messages to be sorted. 
            reverse (bool): If True, sort the messages in descending order. Otherwise, sort the messages in ascending order.
        """
        return sorted(messages, key=lambda msg: datetime.strptime(msg.timestamp, "%Y-%m-%d %H:%M:%S"), reverse=reverse)

    @classmethod
    def sort(cls, messages: List['Message'], key: Optional[Callable[['Message'], Any]] = None, reverse: bool = False) -> List['Message']:
        """
        sort the messages using key or timestamp (by default). 

        Args:
            messages (List[Message]): the messages to be sorted. 
            key (Optional[Callable[['Message'], Any]]): the function used to sort messages. 
            reverse (bool): If True, sort the messages in descending order. Otherwise, sort the messages in ascending order.
        """
        if key is None:
            return cls.sort_by_timestamp(messages, reverse=reverse)
        return sorted(messages, key=key, reverse=reverse)

    @classmethod
    def merge(cls, messages: List[List['Message']], sort: bool=False, key: Optional[Callable[['Message'], Any]] = None, reverse: bool=False) -> List['Message']:
        """
        merge different message list. 

        Args:
            messages (List[List[Message]]): the message lists to be merged. 
            sort (bool): whether to sort the merged messages.
            key (Optional[Callable[['Message'], Any]]): the function used to sort messages. 
            reverse (bool): If True, sort the messages in descending order. Otherwise, sort the messages in ascending order.
        """
        merged_messages = sum(messages, [])
        if sort:
            merged_messages = cls.sort(merged_messages, key=key, reverse=reverse)
        return merged_messages
    

