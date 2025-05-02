from pydantic import Field
from typing import Optional, Any, Callable
import json
from ..core.logging import logger
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput
from ..core.message import Message
from typing import List
from ..prompts.tool_caller import TOOL_CALLER_PROMPT
from ..tools.tool import Tool

class ToolCallingInput(ActionInput):
    query: str = Field(description="The query that might need to use tools to answer")

class ToolGeneratingOutput(ActionOutput):
    function_params: list[dict[str, Any]] = Field(default_factory=list, description="Parameters to pass to the callable")
    continue_after_tool_call: bool = Field(description="Whether to continue the conversation after the tool call")
    
class ToolCallingOutput(ActionOutput):
    content: str = Field(description="The answer to the query")


class ToolCalling(Action):
    max_tool_try: int = 2
    tools_schema: Optional[dict] = None
    tools_caller: Optional[dict[str, Callable]] = None
    conversation: Optional[Message] = None
    tool_generating_output_format: Optional[Any] = None

    def __init__(self, **kwargs):
        name = kwargs.pop("name", "tool_calling")
        description = kwargs.pop("description", "Call a tool function and return the results")
        inputs_format = kwargs.pop("inputs_format", ToolCallingInput)
        outputs_format = kwargs.pop("outputs_format", ToolCallingOutput)
        super().__init__(name=name, description=description, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
        self.tool_generating_output_format = kwargs.pop("intermediate_output_format", ToolGeneratingOutput)
        # Allow max_tool_try to be configured through kwargs
        self.max_tool_try = kwargs.pop("max_tool_try", 2)
    
    def add_tools(self, tools: List[Tool]):
        tools_schemas = [tool.get_tool_schemas() for tool in tools]
        tools_schemas = [j for i in tools_schemas for j in i]
        tools_callers = [tool.get_tools() for tool in tools]
        tools_callers = [j for i in tools_callers for j in i]
        tools_names = [i["function"]["name"] for i in tools_schemas]
        if not self.tools_schema:
            self.tools_schema = {}
            self.tools_caller = {}
        for tool_schema, tool_caller, tool_name in zip(tools_schemas, tools_callers, tools_names):
            self.tools_schema[tool_name] = tool_schema
            self.tools_caller[tool_name] = tool_caller
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, time_out = 0, history: Optional[List[Message]] = None, **kwargs) -> ToolCallingOutput:
        if not inputs:
            logger.error("ToolCalling action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to ToolCalling action is None or empty.')

        time_out += 1
        if time_out > self.max_tool_try:
            if return_prompt:
                prompt_params_values = inputs.get("query")
                return {"content": inputs["query"] + f"\n\nTool execution passing max depth: {self.max_tool_try}"}, prompt_params_values
            return {"content": inputs["query"] + f"\n\nTool execution passing max depth: {self.max_tool_try}"}
        
        print("_______________________ Start Tool Calling _______________________")
        ## 1. Generate tool call args
        prompt_params_values = inputs.get("query")
        
        # Make sure we use the provided system message
        system_message = sys_msg if sys_msg else TOOL_CALLER_PROMPT["system_prompt"]
        history = history if history else []
        
        # Convert Message history to OpenAI format
        messages = []
        
        # Add system message
        messages.append({"role": "system", "content": system_message})
        
        # Add history messages in OpenAI format
        for msg in history:
            content = str(msg.content)
            # Skip empty messages
            if not content.strip():
                continue
                
            if msg.agent == "user":
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})
        
        # Add the current query as a user message
        messages.append({"role": "user", "content": prompt_params_values})
        
        # Remove history from kwargs to avoid passing it as an unknown param
        kwargs_copy = kwargs.copy()
        if 'history' in kwargs_copy:
            del kwargs_copy['history']
        
        tool_call_args = llm.generate(
            messages=messages,
            parser=self.tool_generating_output_format,
            history = history,
            parse_mode="json"
        )
        
        print("Tool call args:")
        print(tool_call_args)
        
        ## 2. Call the tools
        function_params = tool_call_args.function_params
        
        errors = []
        results  =[]
        for function_param in function_params:
            try:
                function_name = function_param.get("function_name")
                function_args = function_param.get("function_args") or {}
        
                # Check if we have a valid function to call
                if not function_name:
                    errors.append("No function name provided")
                    break
                
                print(self.tools_caller)
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
                    print("_____________________ Start Function Calling _____________________")
                    print(f"Executing function calling: {function_name} with {function_args}")
                    result = callable_fn(**function_args)
                        
                except Exception as e:
                    logger.error(f"Error executing tool {function_name}: {e}")
                    errors.append(f"Error executing tool {function_name}: {str(e)}")
                    break
            
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing tool: {e}")
                errors.append(f"Error executing tool: {str(e)}")
        

        ## 3. Add the tool call results to the query and continue the conversation
        results = {"result": results, "error": errors}
        
        inputs = inputs.copy()
        
        ##### ___________ Custom Object Serializer ___________
        # Define a custom object   that can handle various types
        def object_serializer(obj):
            # Handle Pydantic models
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            # Handle objects with __dict__ attribute
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            # Handle objects with custom __str__ method
            elif hasattr(obj, "__str__"):
                return str(obj)
            # Default fallback
            else:
                return repr(obj)
        
        
        ##### ___________ Converting results to string ___________
        try:
            print("Serializing results...")
            results_str = json.dumps(results, default=object_serializer)
        except TypeError as e:
            # If JSON serialization fails, use a more direct approach
            print(f"JSON serialization failed: {e}")
            
            # Build a more controlled string representation
            result_parts = []
            result_parts.append("Results:")
            
            # Process the actual results
            for i, res in enumerate(results.get("result", [])):
                result_parts.append(f"  Result {i+1}: {object_serializer(res)}")
                
            # Process any errors
            for i, err in enumerate(results.get("error", [])):
                result_parts.append(f"  Error {i+1}: {err}")
                
            # Join all parts with newlines
            results_str = "\n".join(result_parts)
            
        ##### ___________ Adding tool call results to the query ___________
        inputs["query"] += "\n\n ### Tool Call Results \n\n" + results_str
        
        print("\nContinue after tool call? :")
        print(tool_call_args.continue_after_tool_call)
        if tool_call_args.continue_after_tool_call:
            # Only continue if we haven't exceeded max_tool_try
            if time_out < self.max_tool_try:
                print(f"Continuing with tool call execution (attempt {time_out+1}/{self.max_tool_try})")
                content = self.execute(llm, inputs, sys_msg, return_prompt, **kwargs, time_out=time_out)
            else:
                print(f"Maximum tool call depth ({self.max_tool_try}) reached, stopping execution")
                content = {"answer": inputs["query"] + f"\n\nTool execution reached maximum depth ({self.max_tool_try})."}
        else:
            content = {"answer": inputs["query"]}
            
        print("answer:")
        print(content)
        
        
        if return_prompt:
            return content, prompt_params_values
        return content
    

