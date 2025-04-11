from typing import Optional, Dict, Any, List

from .action import Action, ActionInput, ActionOutput
from .tool_caller_parser import ToolCallerParser
from ..models.base_model import BaseLLM
from ..core.logging import logger
from ..prompts.tool_caller import TOOL_CALLER_PROMPT
from pydantic import Field


class ToolCallerActionInput(ActionInput):
    query: str = Field(description="The user query to be analyzed")
    tool_descriptions: List[Dict[str, Any]] = Field(description="List of tool descriptions with name, description, and input schema")


class ToolCallerAction(Action):
    """
    Action that analyzes a query and determines whether to use a tool or provide a direct answer.
    """
    
    def __init__(self, **kwargs):
        name = kwargs.pop("name", "tool_caller_action")
        description = kwargs.pop("description", TOOL_CALLER_PROMPT["description"])
        inputs_format = kwargs.pop("inputs_format", ToolCallerActionInput)
        outputs_format = kwargs.pop("outputs_format", ToolCallerParser)
        prompt = kwargs.pop("prompt", TOOL_CALLER_PROMPT["prompt"])
        system_prompt = kwargs.pop("system_prompt", TOOL_CALLER_PROMPT["system_prompt"])
        super().__init__(
            name=name, 
            description=description,
            inputs_format=inputs_format, 
            outputs_format=outputs_format, 
            prompt=prompt,
            **kwargs
        )
        self.system_prompt = system_prompt
    
    def _format_tool_descriptions(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format tool descriptions for the prompt
        
        Args:
            tools: List of tool descriptions
            
        Returns:
            Formatted string of tool descriptions
        """
        formatted_descriptions = []
        
        for tool in tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            
            # Handle both standard format and MCP format
            input_schema = tool.get("input_schema", {})
            
            # If input_schema isn't a dict, it might be the direct schema object
            if not isinstance(input_schema, dict):
                input_schema = {}
                
            # Check if this is an MCP tool schema
            if not input_schema and "function" in tool:
                # Extract from MCP format
                function_info = tool.get("function", {})
                name = function_info.get("name", name)
                description = function_info.get("description", description)
                input_schema = function_info.get("parameters", {})
                
            # Get properties and required fields
            properties = input_schema.get("properties", {})
            required_params = input_schema.get("required", [])
            
            params_str = ""
            if properties:
                param_descriptions = []
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    required = "required" if param_name in required_params else "optional"
                    param_descriptions.append(f"  - {param_name}: {param_type} ({required}) - {param_desc}")
                
                params_str = "\n".join(param_descriptions)
            
            tool_str = f"### {name}\n{description}\n\n**Parameters:**\n{params_str}\n"
            formatted_descriptions.append(tool_str)
        
        return "\n".join(formatted_descriptions)
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str] = None, return_prompt: bool = False, **kwargs) -> ToolCallerParser:
        """
        Execute the tool caller action
        
        Args:
            llm: The language model to use
            inputs: Action inputs including query and tool descriptions
            sys_msg: Optional system message override
            return_prompt: Whether to return the prompt with the output
            
        Returns:
            ToolCallerParser containing the action decision
        """
        if not inputs:
            logger.error("ToolCallerAction received invalid inputs: None or empty")
            raise ValueError("The inputs to ToolCallerAction is None or empty")
        
        query = inputs.get("query")
        tool_descriptions = inputs.get("tool_descriptions", [])
        
        if not query:
            logger.error("ToolCallerAction missing required input: query")
            raise ValueError("Missing required input: query")
        
        # Format the tool descriptions for the prompt
        formatted_tools = self._format_tool_descriptions(tool_descriptions)
        
        # Prepare the prompt variables
        prompt_vars = {
            "query": query,
            "tool_descriptions": formatted_tools
        }
        
        # Use the system prompt from the action or the override
        system_message = sys_msg if sys_msg is not None else self.system_prompt
        
        # Generate the response using the LLM
        llm_response = llm.generate(
            prompt=self.prompt.format(**prompt_vars),
            system_message=system_message
        )
        
        # Parse the response to determine the action
        parser = ToolCallerParser.from_str(llm_response.content)
        
        if return_prompt:
            prompt = self.prompt.format(**prompt_vars)
            return parser, prompt
        
        return parser