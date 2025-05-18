from typing import Optional, List, Callable
from .tool import Tool

class BaseInterpreter(Tool):
    """
    Base class for interpreter tools that execute code securely.
    Implements the standard tool interface with get_tool_schemas and execute methods.
    """

    def __init__(
        self, 
        name: str = 'Base Interpreter',
        schemas: Optional[List[dict]] = None,
        descriptions: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None, 
        **kwargs
    ):
        # Get default values for required attributes
        name = name or 'Base Interpreter'
        schemas = schemas or self.get_tool_schemas()
        descriptions = descriptions or self.get_tool_descriptions()
        tools = tools or self.get_tools()
        
        # Pass these to the parent class initialization
        super().__init__(
            name=name,
            schemas=schemas,
            descriptions=descriptions,
            tools=tools,
            **kwargs
        )

