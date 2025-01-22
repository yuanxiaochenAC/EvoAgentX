from pydantic import Field
from typing import Optional, List

from ..core.module import BaseModule
from ..core.module_utils import generate_id
from ..core.message import Message
from ..core.registry import MODEL_REGISTRY
from ..models.model_configs import LLMConfig
from ..models.base_model import BaseLLM
from ..memory.memory import ShortTermMemory
from ..memory.long_term_memory import LongTermMemory
from ..memory.memory_manager import MemoryManager
from ..storages.base import StorageHandler
from ..actions.action import Action


class Agent(BaseModule):

    name: str # should be unique
    description: str
    llm_config: Optional[LLMConfig] = None
    llm: Optional[BaseLLM] = None
    agent_id: Optional[str] = Field(default_factory=generate_id)
    system_prompt: Optional[str] = None
    short_term_memory: Optional[ShortTermMemory] = Field(default_factory=ShortTermMemory) # store short term memory for a single workflow.
    use_long_term_memory: Optional[bool] = False
    storage_handler: Optional[StorageHandler] = None
    long_term_memory: Optional[LongTermMemory] = None
    long_term_memory_manager: Optional[MemoryManager] = None
    actions: List[Action] = Field(default=None)
    is_human: bool = Field(default=False)
    version: int = 0 

    def init_module(self):
        if not self.is_human:
            assert self.llm_config or self.llm, "must provide either ``llm_config`` or ``llm`` when is_human=False"
            if self.llm_config and not self.llm:
                llm_cls = MODEL_REGISTRY.get_model(self.llm_config.llm_type)
                self.llm = llm_cls(config=self.llm_config)
            if self.llm:
                self.llm_config = self.llm.config
        if self.use_long_term_memory:
            assert self.storage_handler is not None, "must provide ``storage_handler`` when use_long_term_memory=True"
            # TODO revise the initialisation of long_term_memory and long_term_memory_manager
            if not self.long_term_memory:
                self.long_term_memory = LongTermMemory()
            if not self.long_term_memory_manager:
                self.long_term_memory_manager = MemoryManager(
                    storage_handler=self.storage_handler,
                    memory=self.long_term_memory
                )
        self._save_ignore_fields = ["llm"]

    def execute(self, action_name: str, msgs: List[Message], **kwargs) -> Message:
        """
        Execute an action.

        Args:
            action_name (str): the name of the action to execute. 
            msgs (List[Message]): the context for the current task. 
        
        Returns:
            Message: a message that contains the execution results. 
        """
        pass 

    def clear_short_term_memory(self):
        """
        remove all the content in the short term memory.
        """
        pass 
        
    def __eq__(self, other: "Agent"):
        return self.agent_id == other.agent_id

    def __hash__(self):
        return self.agent_id
        
    def save_module(self, path: str, ignore: List[str] = [], **kwargs)-> str:
        ignore_fields = self._save_ignore_fields + ignore
        super().save_module(path=path, ignore=ignore_fields, **kwargs)
