from pydantic import Field
from typing import Optional, Any, Callable, Dict
import json
from ..core.logging import logger
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput
from ..core.message import Message
from ..prompts.tool_caller import TOOL_CALLER_PROMPT

SUMMARIZING_PROMPT = """
You are a helpful assistant that summarizes the results of tool calls.
You may judge whether the problem is solved. If it is not solved, you may explain why.

The result should be in the following format:

{
    "summary": "The summary of the tool call"
}

"""

class ToolCallSummarizingInput(ActionInput):
    query: str = Field(description="The query to generate a tool call for")

class ToolCallSummarizingOutput(ActionOutput):
    summary: str = Field(description="A summary of the tool call")
    
class ToolCallSummarizing(Action):
    
    def __init__(self, **kwargs):
        name = kwargs.pop("name", "tool_call_summarizing")
        description = kwargs.pop("description", "Summarize the results of tool call against the query")
        inputs_format = kwargs.pop("inputs_format", ToolCallSummarizingInput)
        outputs_format = kwargs.pop("outputs_format", ToolCallSummarizingOutput)
        super().__init__(name=name, description=description, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
    
    async def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> ToolCallSummarizingOutput:
        if not inputs:
            logger.error("ToolCalling action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to ToolCalling action is None or empty.')

        print("_______________________ Start Tool Calling _______________________")
        ## 1. Generate tool call args
        prompt_params_values = inputs.get("query")
        output = llm.generate(
            prompt = prompt_params_values, 
            system_message = SUMMARIZING_PROMPT, 
            parser=self.outputs_format,
            parse_mode="json"
        )
        
        print(output)
        
        if return_prompt:
            return output, prompt_params_values
        
        return output


class ToolCallingInput(ActionInput):
    query: str = Field(description="The query that might need to use tools to answer")

class ToolGeneratingOutput(ActionOutput):
    function_params: list[dict[str, Any]] = Field(default_factory=list, description="Parameters to pass to the callable")
    continue_after_tool_call: bool = Field(description="Whether to continue the conversation after the tool call")
    
class ToolCallingOutput(ActionOutput):
    answer: str = Field(description="The answer to the query")


class ToolCalling(Action):
    tools_schema: Optional[dict] = None
    tools_caller: Optional[dict[str, Callable]] = None
    conversation: Optional[Message] = None
    tool_generating_output_format: Optional[Any] = None
    max_tool_try: int = 6

    def __init__(self, **kwargs):
        name = kwargs.pop("name", "tool_calling")
        description = kwargs.pop("description", "Call a tool function and return the results")
        inputs_format = kwargs.pop("inputs_format", ToolCallingInput)
        outputs_format = kwargs.pop("outputs_format", ToolCallingOutput)
        super().__init__(name=name, description=description, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
        self.tool_generating_output_format = kwargs.pop("intermediate_output_format", ToolGeneratingOutput)
    
    def add_tool(self, tools_schema: dict, tools_caller: Callable):
        if self.tools_schema is None:
            self.tools_schema = {}
            self.tools_caller = {}
        self.tools_schema[tools_schema["name"]] = tools_schema
        self.tools_caller[tools_schema["name"]] = tools_caller
    
    async def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, time_out = 0, **kwargs) -> ToolCallingOutput:
        if not inputs:
            logger.error("ToolCalling action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to ToolCalling action is None or empty.')

        if time_out >= self.max_tool_try:
            return inputs["query"] + f"Tool executino passing max deepth: {self.max_tool_try}"
        
        print("_______________________ Start Tool Calling _______________________")
        ## 1. Generate tool call args
        prompt_params_values = inputs.get("query")
        tool_call_args = llm.generate(
            prompt = prompt_params_values, 
            system_message = sys_msg, 
            parser=self.tool_generating_output_format,
            parse_mode="json"
        )
        
        print("Tool call args:")
        print(tool_call_args)
        
        ## 2. Call the tools
        function_params = tool_call_args.function_params
        
        errors = []
        results  =[]
        for function_param in function_params:
            function_name = function_param.get("function_name")
            function_args = function_param.get("function_args") or {}
        
            # Check if we have a valid function to call
            if not function_name:
               errors.append("No function name provided")
               break
                
            # Try to get the callable from our tools_caller dictionary
            callable_fn = None
            if self.tools_caller and function_name in self.tools_caller:
                callable_fn = self.tools_caller[function_name]
            elif callable(function_name):
                callable_fn = function_name
                
            if not callable_fn:
                errors.append(f"Function '{function_name}' not found or not callable")
                break
            
            try:
                # Determine if the function is async or not
                import inspect
                if inspect.iscoroutinefunction(callable_fn):
                    # Handle async function
                    result = await callable_fn(**function_args)
                else:
                    # Handle regular function
                    result = callable_fn(**function_args)
                    
            except Exception as e:
                logger.error(f"Error executing tool {function_name}: {e}")
                errors.append(f"Error executing tool {function_name}: {str(e)}")
                break
        
            results.append(result)

        ## 3. Add the tool call results to the query and continue the conversation
        results = {"result": results, "error": errors}
        
        inputs["query"] += "\n\n ### Tool Call Results \n\n" + str(results)
        
        print("\n\n\n\n\n\n\n\nContinue after tool call? :")
        print(tool_call_args.continue_after_tool_call)
        if tool_call_args.continue_after_tool_call:
            answer = await self.execute(llm, inputs, sys_msg, return_prompt, **kwargs)
            print(answer)
        else:
            answer = "Tool call results: " + str(results)
        
        if return_prompt:
            return str(answer), prompt_params_values
        
        return str(answer)
        
        
        
