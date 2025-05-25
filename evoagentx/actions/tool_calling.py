from pydantic import Field, create_model
from typing import Optional, Any, Callable, Type, List, Union
import re
import json

from ..core.logging import logger
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput
from ..core.message import Message
from ..prompts.tool_calling import GOAL_BASED_TOOL_CALLING_PROMPT, OUTPUT_EXTRACTION_PROMPT
from ..tools.tool import Tool
from ..utils.utils import generate_dynamic_class_name
from ..core.registry import MODULE_REGISTRY
from ..models.base_model import LLMOutputParser
from ..core.module_utils import parse_json_from_llm_output

class ToolCalling(Action):
    max_tool_try: int = 2
    tools_schema: Optional[dict] = None
    tools_caller: Optional[dict[str, Callable]] = None
    tool_calling_instruction: Optional[str] = None
    conversation: Optional[Message] = None
    inputs: dict = {}
    outputs: dict = {}
    output_parser: Optional[Type[ActionOutput]] = None
    prompt: str = ""
    execution_history: list[Any] = []
    llm: BaseLLM = None
    
    def __init__(self, **kwargs):
        name = kwargs.pop("name", "tool_calling")
        description = kwargs.pop("description", "Call a tool function and return the results")
        inputs_format = kwargs.pop("inputs")
        outputs_format = kwargs.pop("outputs")
        output_parser = kwargs.pop("output_parser", None)
        super().__init__(name=name, description=description, **kwargs)
        self._generate_inputs_outputs_info(inputs_format, outputs_format, output_parser, name)
        # Allow max_tool_try to be configured through kwargs
        self.max_tool_try = kwargs.pop("max_tool_try", 2)
        self.prompt = kwargs.pop("prompt", "")
    
    def _generate_inputs_outputs_info(self, inputs: List[dict], outputs: List[dict], output_parser: ActionOutput = None, name: str = None):
        # create the action input type
        action_input_fields = {}
        for field in inputs:
            required = field.get("required", True)
            if required:
                action_input_fields[field["name"]] = (str, Field(description=field["description"]))
            else:
                action_input_fields[field["name"]] = (Optional[str], Field(default=None, description=field["description"]))
        
        action_input_type = create_model(
            self._get_unique_class_name(
                generate_dynamic_class_name(name+" action_input")
            ),
            **action_input_fields, 
            __base__=ActionInput
        )
        
        # create the action output type
        if output_parser is None:
            action_output_fields = {}
            for field in outputs:
                required = field.get("required", True)
                if required:
                    action_output_fields[field["name"]] = (Union[str, dict, list], Field(description=field["description"]))
                else:
                    action_output_fields[field["name"]] = (Optional[Union[str, dict, list]], Field(default=None, description=field["description"]))
            
            action_output_type = create_model(
                self._get_unique_class_name(
                    generate_dynamic_class_name(name+" action_output")
                ),
                **action_output_fields, 
                __base__=ActionOutput,
                # get_content_data=customize_get_content_data,
                # to_str=customize_to_str
            )
        else:
            # self._check_output_parser(outputs, output_parser)
            self.action_output_type = output_parser
        
        self.inputs_format = action_input_type
        self.outputs_format = action_output_type
    
    def _get_unique_class_name(self, candidate_name: str) -> str:
        """
        Get a unique class name by checking if it already exists in the registry.
        If it does, append "Vx" to make it unique.
        """
        if not MODULE_REGISTRY.has_module(candidate_name):
            return candidate_name 
        
        i = 1 
        while True:
            unique_name = f"{candidate_name}V{i}"
            if not MODULE_REGISTRY.has_module(unique_name):
                break
            i += 1 
        return unique_name 
    
    def add_tools(self, tools: List[Tool]):
        tools_schemas = [tool.get_tool_schemas() for tool in tools]
        tools_schemas = [j for i in tools_schemas for j in i]
        tools_callers = [tool.get_tools() for tool in tools]
        tools_callers = [j for i in tools_callers for j in i]
        tools_names = [i["function"]["name"] for i in tools_schemas]
        self.tool_calling_instructions = [tool.get_tool_prompt() for tool in tools]
        if not self.tools_schema:
            self.tools_schema = {}
            self.tools_caller = {}
        for tool_schema, tool_caller, tool_name in zip(tools_schemas, tools_callers, tools_names):
            self.tools_schema[tool_name] = tool_schema
            self.tools_caller[tool_name] = tool_caller
    
    def _extract_tool_calls(self, llm_output: str):
        if match := re.search(r"```(?:ToolCalling)?\s*\n(.*?)\n```", llm_output, re.DOTALL):
            json_str = match.group(1)
            return json.loads(json_str)
        return None
    
    def _extract_output(self, llm_output: Any, llm: BaseLLM = None, **kwargs):
        attr_descriptions: dict = self.outputs_format.get_attr_descriptions()
        output_description_list = [] 
        for i, (name, desc) in enumerate(attr_descriptions.items()):
            output_description_list.append(f"{i+1}. {name}\nDescription: {desc}")
        output_description = "\n\n".join(output_description_list)
        extraction_prompt = self.prompt + "\n\n" + OUTPUT_EXTRACTION_PROMPT.format(text=llm_output, output_description=output_description)
        llm_extracted_output: LLMOutputParser = llm.generate(prompt=extraction_prompt, history=kwargs.get("history", []) + [llm_output])
        llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
        output = self.outputs_format.from_dict(llm_extracted_data)
        
        print("Extracted output:")
        print(output)
        
        return output
    
    def _calling_tools(self, tool_call_args) -> dict:
        ## ___________ Call the tools ___________
        errors = []
        results  =[]
        for function_param in tool_call_args:
            try:
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
        return results
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, return_prompt: bool = False, time_out = 0, **kwargs):
        if not inputs:
            logger.error("ToolCalling action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to ToolCalling action is None or empty.')

        self.execution_history = []
        
        print("_______________________ Start Tool Calling _______________________")
        ## 1. Generate tool call args
        input_attributes: dict = self.inputs_format.get_attr_descriptions()
        prompt_params_values = {k: inputs[k] for k in input_attributes.keys()}
        
        while True:
            ### Generate response from LLM
            if time_out > self.max_tool_try:
                if return_prompt:
                    return self._extract_output("{content}".format(content = self.execution_history), llm = llm), self.prompt
                return self._extract_output("{content}".format(content = self.execution_history), llm = llm) 
            
            tool_call_args = llm.generate(
                prompt = GOAL_BASED_TOOL_CALLING_PROMPT.format(
                    goal_prompt = self.prompt, 
                    inputs = prompt_params_values, 
                    history = self.execution_history, 
                    tools_description = self.tools_schema, 
                    additional_context = self.tool_calling_instructions
                ), 
                parse_mode="json"
            )
            
            print("Tool call args:")
            print(tool_call_args)
            
            tool_call_args = self._extract_tool_calls(tool_call_args.content)
            if not tool_call_args:
                break
            
            print("Extracted tool call args:")
            print(tool_call_args)
            
            results = self._calling_tools(tool_call_args)
            
            print("results:")
            print(results)
            
            self.execution_history.append({"tool_call_args": tool_call_args, "results": results})
        
        if return_prompt:
            return self._extract_output("{content}".format(content = self.execution_history), llm = llm), self.prompt
        return self._extract_output("{content}".format(content = self.execution_history), llm = llm) 
        

