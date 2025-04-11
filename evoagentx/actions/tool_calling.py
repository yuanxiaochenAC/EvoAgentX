from pydantic import Field
from typing import Optional, Dict, Any, Callable

from ..core.logging import logger
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput


class ToolCallingInput(ActionInput):
    callable_fn: Callable = Field(description="The callable function object to execute")
    function_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the callable")


class ToolCallingOutput(ActionOutput):
    result: Any = Field(description="Result from the tool execution")
    error: Optional[str] = Field(default=None, description="Error message if the tool execution failed")


class ToolCalling(Action):

    def __init__(self, **kwargs):
        name = kwargs.pop("name", "tool_calling")
        description = kwargs.pop("description", "Call a tool function and return the results")
        inputs_format = kwargs.pop("inputs_format", ToolCallingInput)
        outputs_format = kwargs.pop("outputs_format", ToolCallingOutput)
        super().__init__(name=name, description=description, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
        
    async def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> ToolCallingOutput:
        if not inputs:
            logger.error("ToolCalling action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to ToolCalling action is None or empty.')

        callable_fn = inputs.get("callable_fn")
        function_args = inputs.get("function_args") or {}
        
        if not callable_fn or not callable(callable_fn):
            return ToolCallingOutput(
                result=None,
                error="No valid callable function provided"
            )
        
        try:
            # Determine if the function is async or not
            import inspect
            if inspect.iscoroutinefunction(callable_fn):
                # Handle async function
                result = await callable_fn(**function_args)
            else:
                # Handle regular function
                result = callable_fn(**function_args)
                
            return ToolCallingOutput(result=result)
        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            return ToolCallingOutput(
                result=None,
                error=f"Error executing tool: {str(e)}"
            )
