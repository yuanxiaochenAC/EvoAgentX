from enum import Enum
from typing import List
from ..core.module import BaseModule
from ..core.message import Message


class TrajectoryState(str, Enum):
    """
    Enum representing the status of a trajectory step.
    """
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrajectoryStep(BaseModule):

    message: Message = None
    error: str = None
    status: TrajectoryState


class Environment(BaseModule):

    """
    Responsible for storing and managing intermediate states of execution.
    """
    trajectory: List[TrajectoryStep] = []

    def publish_message(self, message: Message, state: TrajectoryState = None, error: str = None, **kwargs):
        """
        Add a message to the shared memory and optionally to a specific task's message list.

        Args:
            message (Message): The message to be added.
            task_name (str, optional): The name of the task this message is related to. If None, the message is considered global.
        """
        pass 

    def get_task_messages(self, task_name: str, **kwargs) -> List[Message]:
        """
        Retrieve all messages related to a specific task.

        Args:
            task_id (str): The ID of the task.

        Returns:
            List[Message]: A list of messages related to the task.
        """
        pass 

    def clear_task_messages(self, task_name: str, **kwargs):
        """
        Clear all messages related to a specific task.

        Args:
            task_id (str): The ID of the task.
        """
        pass 

    def clear_all_messages(self):
        """
        Clear all messages, both shared and task-specific.
        """
        pass 


