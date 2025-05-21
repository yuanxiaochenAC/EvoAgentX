from .tool import Tool

class BaseInterpreter(Tool):
    """
    Base class for interpreter tools that execute code securely.
    Implements the standard tool interface with get_tool_schemas and execute methods.
    """

    def __init__(
        self, 
        name: str = 'BaseInterpreter',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

