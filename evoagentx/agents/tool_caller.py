from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field

from .agent import Agent
from ..actions.tool_calling import ToolCalling
from ..prompts.tool_caller import TOOL_CALLER_PROMPT
from ..core.message import Message, MessageType


class ToolCaller(Agent):
    """
    An agent that determines whether to call a tool or provide a direct answer
    based on analyzing the user query and available tools.
    
    This agent has one main action:
    1. ToolCalling - Executes the selected tool if required
    """
    
    tool_descriptions: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of available tools with their descriptions"
    )
    
    def __init__(self, **kwargs):
        name = kwargs.pop("name", TOOL_CALLER_PROMPT["name"])
        description = kwargs.pop("description", TOOL_CALLER_PROMPT["description"])
        system_prompt = kwargs.pop("system_prompt", TOOL_CALLER_PROMPT["system_prompt"])
        
        # Initialize actions
        tool_calling_action = ToolCalling()
        actions = kwargs.pop("actions", [tool_calling_action])
        
        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt, 
            actions=actions,
            **kwargs
        )
        
        # Store action name for convenience
        self.tool_calling_action_name = tool_calling_action.name
    
    def add_tool_descriptions(self, tools: List[Dict[str, Any]]):
        """
        Add tool descriptions to the agent's tool list
        
        Args:
            tools: List of tool descriptions with name, description, and input schema
        """
        self.tool_descriptions.extend(tools)
    
    def add_tools_from_toolkit(self, toolkit_tools: List[Tuple[callable, Dict[str, Any]]]):
        """
        Add tools from a toolkit that returns function and schema pairs
        
        Args:
            toolkit_tools: List of tuples with (function, schema) pairs
        """
        for tool_func, tool_schema in toolkit_tools:
            tool_name = tool_schema["function"]["name"]
            tool_description = tool_schema["function"]["description"]
            tool_parameters = tool_schema["function"]["parameters"]
            
            self.tool_descriptions.append({
                "name": tool_name,
                "description": tool_description,
                "callable_fn": tool_func,
                "input_schema": tool_parameters
            })
    
    def clear_tool_descriptions(self):
        """
        Clear all tool descriptions
        """
        self.tool_descriptions = []
    
    async def execute(self, action_name: str, action_input_data: dict, return_msg_type: MessageType = MessageType.UNKNOWN) -> Message:
        """
        Execute an action with the given input data
        
        Args:
            action_name: Name of the action to execute
            action_input_data: Input data for the action
            return_msg_type: Message type for the returned message
            
        Returns:
            Message containing the execution results
        """
        # Execute the action using the base class's execute method
        message = await super().execute(action_name, action_input_data, return_msg_type)
        
        # Handle tool call if needed
        if message.content.action == "tool_call":
            tool_call_info = message.content.get_tool_call_info()
            if not tool_call_info:
                return message
                
            tool_name = tool_call_info.get("tool_name")
            tool_parameters = tool_call_info.get("parameters", {})
            
            # Find the matching tool
            matching_tool = next(
                (tool for tool in self.tool_descriptions 
                 if tool.get("name") == tool_name),
                None
            )
            
            if not matching_tool:
                message.content = f"Error: Tool '{tool_name}' not found in available tools"
                return message
                
            if not matching_tool.get("callable_fn"):
                message.content = f"Error: Tool '{tool_name}' has no callable function"
                return message
            
            # Execute the tool call
            tool_input_data = {
                "callable_fn": matching_tool["callable_fn"],
                "function_args": tool_parameters
            }
            
            try:
                tool_result = await super().execute(
                    self.tool_calling_action_name,
                    tool_input_data,
                    MessageType.TOOL_RESULT
                )
                
                # Check if the tool execution was successful
                if hasattr(tool_result.content, 'error') and tool_result.content.error:
                    message.content = f"Error executing tool '{tool_name}': {tool_result.content.error}"
                else:
                    message = tool_result
                    
            except Exception as e:
                message.content = f"Error executing tool '{tool_name}': {str(e)}"
        
        return message