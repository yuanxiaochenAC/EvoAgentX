from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field

from .agent import Agent
from ..actions.tool_calling import ToolCalling, ToolCallSummarizing
from ..prompts.tool_caller import TOOL_CALLER_PROMPT
from ..core.message import Message, MessageType
from ..tools.mcp import MCPToolkit
from typing import Type, Optional, Union, Tuple, List, Callable
from ..core.module_utils import parse_json_from_text


class ToolCaller(Agent):
    """
    An agent that determines whether to call a tool or provide a direct answer
    based on analyzing the user query and available tools.
    
    This agent has one main action:
    1. ToolCalling - Executes the selected tool if required
    """
    tools_schema: Optional[List[dict]] = None
    tools_caller: Optional[dict[str, Callable]] = None
    mcp_toolkit: Optional[MCPToolkit] = None
    tool_calling_action: Optional[ToolCalling] = None
    tool_summarizing_action: Optional[ToolCallSummarizing] = None
    
    def __init__(self, **kwargs):
        name = kwargs.pop("name", TOOL_CALLER_PROMPT["name"])
        description = kwargs.pop("description", TOOL_CALLER_PROMPT["description"])
        system_prompt = kwargs.pop("system_prompt", TOOL_CALLER_PROMPT["system_prompt"])
        actions = kwargs.pop("actions", [ToolCalling(), ToolCallSummarizing()])
        
        super().__init__(name=name, description=description, system_prompt=system_prompt, actions=actions, **kwargs)
        self.tool_calling_action = self._action_map[self.tool_calling_action_name]
    
    @property
    def tool_calling_action_name(self):
        return self.get_action_name(action_cls=ToolCalling)
    
    @property
    def tool_summarizing_action_name(self):
        return self.get_action_name(action_cls=ToolCallSummarizing)
    
    def add_tool(self, tool_schema: dict, tools_caller: Callable):
        if self.tools_schema is None:
            self.tools_schema = {}
            self.tools_caller = {}
        self.tools_schema[tool_schema["name"]] = tool_schema
        self.tools_caller[tool_schema["name"]] = tools_caller
        self.tool_calling_action.add_tool(tool_schema, tools_caller)
        self.system_prompt = TOOL_CALLER_PROMPT["prompt"].format(tool_descriptions=self.tools_schema)

    def add_mcp_toolkit(self, mcp_toolkit: MCPToolkit):
        self.mcp_toolkit = mcp_toolkit
        if not mcp_toolkit.is_connected():
            mcp_toolkit.connect()
        tools = mcp_toolkit.get_tools()
        for tool in tools:
            self.add_tool(tool[1]["function"], tool[0])
    
    async def execute(self, **kwargs) -> Message:
        return await super().execute_async(**kwargs)
    