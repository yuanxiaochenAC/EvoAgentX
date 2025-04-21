from typing import List, Dict, Any, Optional, Tuple, Callable
from pydantic import Field
import asyncio

from .customize_agent import CustomizeAgent
from ..actions.tool_calling import ToolCalling
from ..prompts.tool_caller import TOOL_CALLER_PROMPT, CUSTOM_TOOL_CALLER_PROMPT
from ..core.message import Message, MessageType
from ..tools.mcp import MCPToolkit
from ..core.module_utils import parse_json_from_text


class CusToolCaller(CustomizeAgent):
    """
    An agent that determines whether to call a tool or provide a direct answer
    based on analyzing the user query and available tools.
    
    This agent has two main actions:
    1. ToolCalling - Executes the selected tool if required
    """
    tools_schema: Optional[Dict[str, dict]] = None
    tools_caller: Optional[Dict[str, Callable]] = None
    mcp_toolkit: Optional[MCPToolkit] = None
    tool_calling_action: Optional[ToolCalling] = None
    _initialized: bool = False
    ori_prompt: str = ""
    mcp_prompt: str = ""
    
    def __init__(self, **kwargs):
        # Extract our specific parameters before passing to parent
        max_tool_try = kwargs.pop("max_tool_try", 1)
        mcp_config_path = kwargs.pop("mcp_config_path", None)
        
        # Create actions with the max_tool_try value
        # Don't add to kwargs - let CustomizeAgent handle action creation
        tool_actions = [ToolCalling(max_tool_try=max_tool_try)]
        
        # Initialize as CustomizeAgent - all other parameters should be passed by the caller
        super().__init__(**kwargs)
        
        # Now add our tool actions (after CustomizeAgent has created its action)
        self.ori_prompt = self.system_prompt
        self.mcp_prompt = self.ori_prompt + CUSTOM_TOOL_CALLER_PROMPT
        for action in tool_actions:
            self.add_action(action)
        
        # Store values as instance attributes after parent initialization
        self.max_tool_try = max_tool_try
        self.mcp_config_path = mcp_config_path
        
        # Store the tool calling action for easy access
        self.tool_calling_action = self._action_map[self.tool_calling_action_name]
        
        # Initialize MCP toolkit if a config path is provided
        if self.mcp_config_path:
            self._init_mcp_toolkit_from_path(self.mcp_config_path)
    
    def _init_mcp_toolkit_from_path(self, config_path: str) -> None:
        """Initialize MCP toolkit from a configuration file path"""
        self.mcp_toolkit = MCPToolkit(config_path=config_path)
        # Just create the toolkit, don't connect yet
        # Connection will happen in the async initialize method
    
    async def initialize(self) -> None:
        """
        Asynchronously initialize the agent, including connecting to MCP toolkit.
        This should be called after creating the agent and before using it.
        """
        if self._initialized:
            print("Agent already initialized")
            return
            
        print(f"Initializing CusToolCaller agent: {self.name}")
        if self.mcp_toolkit:
            try:
                await self.mcp_toolkit.connect()
                tools = self.mcp_toolkit.get_tools()
                for tool in tools:
                    self.add_tool(tool[1]["function"], tool[0])
                self._initialized = True
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise
        
        if self.tools_schema:
            
            self.mcp_prompt = CUSTOM_TOOL_CALLER_PROMPT + "\n### Tools Available\n" + str(self.tools_schema)
    
    async def cleanup(self) -> None:
        """Clean up resources, particularly disconnecting MCP toolkit"""
        if self.mcp_toolkit and self.mcp_toolkit.is_connected():
            try:
                print(f"Disconnecting MCP toolkit for agent: {self.name}")
                await self.mcp_toolkit.disconnect()
                print(f"MCP toolkit disconnected during cleanup for agent: {self.name}")
            except Exception as e:
                print(f"Error disconnecting MCP toolkit for agent {self.name}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    async def __aenter__(self):
        """Async context manager enter method"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit method"""
        await self.cleanup()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception:
            # Ignore errors during cleanup in destructor
            pass
    
    @property
    def tool_calling_action_name(self) -> str:
        return self.get_action_name(action_cls=ToolCalling)
    
    def add_tool(self, tool_schema: dict, tools_caller: Callable) -> None:
        """Add a tool to the tool caller agent"""
        if self.tools_schema is None:
            self.tools_schema = {}
            self.tools_caller = {}
        
        self.tools_schema[tool_schema["name"]] = tool_schema
        self.tools_caller[tool_schema["name"]] = tools_caller
        self.tool_calling_action.add_tool(tool_schema, tools_caller)
        

    async def add_mcp_toolkit(self, mcp_toolkit: MCPToolkit) -> None:
        """Add MCP toolkit to the tool caller agent"""
        self.mcp_toolkit = mcp_toolkit
        if not mcp_toolkit.is_connected():
            await mcp_toolkit.connect()
        tools = mcp_toolkit.get_tools()
        for tool in tools:
            self.add_tool(tool[1]["function"], tool[0])
        self._initialized = True
    
    async def execute(self, **kwargs) -> Message:
        """Execute the tool caller agent"""
        # Ensure initialization before execution
        # if not self._initialized and self.mcp_toolkit:
        #     await self.initialize()
        # return await super().execute_async(**kwargs)
    
        try:
            print(f"CusToolCaller.execute called for agent: {self.name}")
            print(f"Kwargs: {kwargs}")
            
            if not self._initialized and self.mcp_toolkit:
                print(f"Agent {self.name} not initialized yet, initializing...")
                await self.initialize()
                
            # # Always use the tool_calling_action for CusToolCaller, override the action_name
            action_name = kwargs.get("action_name", self.tool_calling_action_name)
            if action_name == self.tool_calling_action_name:
                self.system_prompt = self.mcp_prompt
                action_input_data = kwargs.get("action_input_data", {})
                if not action_input_data.get("query") and "query" not in action_input_data:
                    # If no query is provided, create one from the task description or goal
                    goal = kwargs.get("wf_goal", "")
                    task_desc = kwargs.get("wf_task_desc", "")
                    if task_desc:
                        action_input_data["query"] = f"Task: {task_desc}\nGoal: {goal}"
                    elif goal:
                        action_input_data["query"] = goal
                        
                kwargs["action_input_data"] = action_input_data
                
            else:
                self.system_prompt = self.ori_prompt
            
            # # Execute using the parent's execute_async method
            result = await super().execute_async(**kwargs)
            print(f"Agent {self.name} execution completed successfully")
            return result
            
        except Exception as e:
            print(f"Error in CusToolCaller.execute for agent {self.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create an error message to return
            from ..core.message import Message, MessageType
            error_message = Message(
                content=f"Error executing agent {self.name}: {str(e)}",
                agent=self.name,
                action=kwargs.get("action_name", self.tool_calling_action_name),
                msg_type=MessageType.ERROR
            )
            return error_message


