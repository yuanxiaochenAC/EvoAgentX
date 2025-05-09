from typing import Dict, Any
from .tool import Tool

class BaseInterpreter(Tool):
    """
    Base class for interpreter tools that execute code securely.
    Implements the standard tool interface with get_tool_schemas and execute methods.
    """

    def __init__(self, **data):
        # Get default values for required attributes
        name = data.pop('name', 'Base Interpreter')
        schemas = data.pop('schemas', self.get_tool_schemas())
        descriptions = data.pop('descriptions', self.get_tool_descriptions())
        tools = data.pop('tools', [])
        
        # Pass these to the parent class initialization
        super().__init__(
            name=name,
            schemas=schemas,
            descriptions=descriptions,
            tools=tools,
            **data
        )

    def get_tool_schemas(self) -> Dict[str, Any]:
        """
        Returns the OpenAI-compatible function schema for the interpreter.
        
        Returns:
            Dict[str, Any]: Function schema in OpenAI format
        """
        pass

    def get_tool_descriptions(self) -> str:
        """
        Returns a brief description of the interpreter tool.
        
        Returns:
            str: Tool description
        """
        pass

    def get_tools(self):
        """
        Returns a list of callable methods provided by this tool.
        
        Returns:
            list: List of callable methods
        """
        pass
