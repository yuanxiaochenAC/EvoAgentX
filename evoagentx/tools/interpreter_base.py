from typing import Dict, Any
from .tool import Tool

class BaseInterpreter(Tool):
    """
    Base class for interpreter tools that execute code securely.
    Implements the standard tool interface with get_tool_schema and execute methods.
    """

    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Returns the OpenAI-compatible function schema for the interpreter.
        
        Returns:
            Dict[str, Any]: Function schema in OpenAI format
        """
        return {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": self.get_tool_description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "The programming language of the code"
                        }
                    },
                    "required": ["code", "language"]
                }
            }
        }

    def get_tool_description(self) -> str:
        """
        Returns a brief description of the interpreter tool.
        
        Returns:
            str: Tool description
        """
        return "BaseInterpreter is an abstract class for interpreters that execute code securely. It enforces safety checks before execution and serves as the foundation for language-specific interpreters."

    def execute(self, code: str, language: str) -> str:
        """
        Checks the code for safety before execution.
        
        Args:
            code (str): The code to execute
            language (str): The programming language of the code
            
        Returns:
            str: Execution result
        """
        raise NotImplementedError("Subclasses must implement execute")
