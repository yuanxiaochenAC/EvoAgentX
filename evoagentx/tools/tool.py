from ..core.module import BaseModule
from typing import Dict, Any, List, Callable

class Tool(BaseModule):
    """
    Base interface for all tools. All tools must implement:
    - get_tool_schema: Returns the OpenAI-compatible function schema
    - execute: Executes the tool with the provided parameters
    """
    name: str
    descriptions: List[str]
    schemas: List[dict[str, Any]]
    tools: List[Callable]

    def get_tool_schemas(self) -> Dict[str, Any]:
        """
        Returns the OpenAI-compatible function schema for this tool.
        The schema follows the format used by MCP servers and OpenAI function calling.
        
        Returns:
            Dict[str, Any]: The function schema in OpenAI format
        """
        if not self.schemas:
            raise NotImplementedError("All tools must implement get_tool_schema")
        return self.schemas

    def get_tools(self) -> List[Callable]:
        """
        Returns a list of callable functions for all tools
        
        Returns:
            List[Callable]: A list of callable functions
        """
        if not self.tools:
            raise NotImplementedError("All tools must implement get_tools")
        return self.tools
    
    # Legacy method - to be deprecated
    def get_tool_descriptions(self) -> List[str]:
        """Legacy method for backward compatibility. Use get_tool_schema instead."""
        if not self.descriptions:
            raise NotImplementedError("All tools must implement get_tool_descriptions")
        return self.descriptions
