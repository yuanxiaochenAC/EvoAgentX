from pydantic import Field, PositiveInt
from typing import Union, List, Dict

from ..core.module import BaseModule
from ..core.module_utils import generate_id, get_timestamp
from ..core.message import Message
# from ..utils.utils import safe_remove


class BaseMemory(BaseModule):

    messages: List[Message] = []
    memory_id: str = Field(default_factory=generate_id)
    timestamp: str = Field(default_factory=get_timestamp)
    capacity: PositiveInt = Field(default=None, description="maximum of messages, None means there is no limit to the message number")

    def init_module(self):
        """
        initialize self._by_sender: Dict[str, List[Message]], self._by_receiver: Dict[str, List[Message]],
        self.by_trigger: Dict[str, List[Message]] from self.messages to faciliate retrieving different types of memory. 
        """
        pass

    @property
    def size(self):
        return len(self.messages)
    
    def clear(self):
        """
        clear all the messages in the memory
        """
        pass
    
    def remove_message(self, message: Message):
        """
        remove a single message.

        Args:
            message (Message): the message to be removed. The message should be deleted from self.messages, self._by_sender, self._by_receiver, self._by_trigger
        """
        pass

    def add_message(self, message: Message):
        """
        store a single message. 

        Args:
            message (Message): the message to be stored. 
        """
        pass
    
    def add_messages(self, messages: Union[Message, List[Message]], **kwargs):
        """
        store (a) message(s) to the memory. 

        Args:
            messages (Union[Message, List[Message]]): the input messages can be a single message or a list of message.
        """
        pass 
    
    def get(self, n: int=None, **kwargs) -> List[Message]:
        """
        return recent messages in the memory. 

        Args: 
            k (int): the number of returned messages. If None, return all the messages in the memory. 
        """
        pass

    def get_by_type(self, data: Dict[str, list], key: str, n: int = None, **kwargs) -> List[Message]:
        """
        Retrieve a list of Message objects from a given data dictionary based on a specified type key.

        This function looks up the value associated with `key` in the `data` dictionary, which should be a list of messages. It then returns a subset of these messages according to the specified parameters.
        If `n` is provided, it limits the number of messages returned; otherwise, it may return the entire list. Additional keyword arguments (**kwargs) can be used to further filter or process the resulting messages.

        Args:
            data (Dict[str, list]): A dictionary where keys are type strings and values are lists of messages.
            key (str): The key in `data` identifying the specific list of messages to retrieve.
            n (int, optional): The maximum number of messages to return. If not provided, all messages under the given `key` may be returned.
            **kwargs: Additional parameters for filtering or processing the messages.

        Returns:
            List[Message]: A list of messages corresponding to the given `key`, possibly filtered or truncated according to `n` and other provided keyword arguments.
        """
        pass
    
    def get_by_sender(self, *senders: str, n: int=None, **kwargs) -> List[Message]:
        """
        return recent messages sent by the sender in the memory. 

        Args:
            sender (str): the sender of the message. 
            n (int): the number of returned messages. 
        """
        pass
    
    def get_by_receiver(self, *receivers: str, n: int=None, **kwargs) -> List[Message]:
        """
        return recent messages sent to the receiver in the memory. 

        Args:
            receiver (str): the receiver of the message. 
            n (int): the number of returned messages. 
        """
        pass

    def get_by_trigger(self, *triggers: str, n: int=None, **kwargs) -> List[Message]:
        """
        return recent messages triggered by trigger in the memory. 

        Args:
            trigger (str): the trigger of the message. 
            n (int): the number of returned messages. 
        """
        pass


class ShortTermMemory(BaseMemory):
    pass


