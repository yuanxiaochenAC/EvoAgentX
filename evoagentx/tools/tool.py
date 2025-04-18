from ..core.module import BaseModule
from typing import Dict, Any, List

class Tool(BaseModule):
    """
    Base interface for all tools. All tools must implement:
    - get_tool_schema: Returns the OpenAI-compatible function schema
    - execute: Executes the tool with the provided parameters
    """

    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Returns the OpenAI-compatible function schema for this tool.
        The schema follows the format used by MCP servers and OpenAI function calling.
        
        Returns:
            Dict[str, Any]: The function schema in OpenAI format
        """
        raise NotImplementedError("All tools must implement get_tool_schema")

    def execute(self, **kwargs) -> Any:
        """
        Executes the tool with the provided parameters.
        
        Args:
            **kwargs: Arguments for the tool execution
            
        Returns:
            Any: Result of the tool execution
        """
        raise NotImplementedError("All tools must implement execute")
        
    # Legacy method - to be deprecated
    def get_tool_info(self) -> dict:
        """Legacy method for backward compatibility. Use get_tool_schema instead."""
        return self.get_tool_schema()
