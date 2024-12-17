from typing import Optional, Type, List

from ..core.module import BaseModule
from ..core.message import Message
from ..models.base_model import BaseLLM
from ..tools.tool import Tool 
from ..core.parser import Parser


class Action(BaseModule):

    name: str
    description: str
    prompt: Optional[str] = None
    tools: Optional[List[Tool]] = None
    parser: Optional[Type[Parser]] = None 

    def init_module(self):
        pass 

    def execute(self, llm: Optional[BaseLLM] = None, msgs: List[Message] = None, sys_msg: Optional[str]=None, **kwargs) -> Parser:
        """
        The main entrance for executing an action. 

        Args:
            llm (BaseLLM, optional): the LLM used to execute the action.
            msgs (List[Message], optional): the context for executing the action.
            sys_msg (str, optional): The system message for the llm.
        
        Returns:
            Parser: returns a Parser object (a structured output of LLM's generated text  or other structured object).
        """
        pass

