import asyncio
import inspect
import json
from pydantic import create_model, Field
from typing import Optional, Callable, Type, List, Any, Union, Dict

from .customize_agent import CustomizeAgent
from ..core.logging import logger
from ..core.registry import MODULE_REGISTRY, ACTION_FUNCTION_REGISTRY
from ..models.model_configs import LLMConfig 
from ..actions.action import Action, ActionOutput, ActionInput
from ..utils.utils import generate_dynamic_class_name, make_parent_folder


class ActionAgent(CustomizeAgent):
    """
    ActionAgent is a specialized agent that executes a provided function directly without LLM.
    It creates an action that uses the provided function as the execution backbone.
    
    Attributes:
        name (str): The name of the agent.
        description (str): A description of the agent's purpose and capabilities.
        inputs (List[dict]): List of input specifications, where each dict contains:
            - name (str): Name of the input parameter
            - type (str): Type of the input
            - description (str): Description of what the input represents
            - required (bool, optional): Whether this input is required (default: True)
        outputs (List[dict]): List of output specifications, where each dict contains:
            - name (str): Name of the output field
            - type (str): Type of the output
            - description (str): Description of what the output represents
            - required (bool, optional): Whether this output is required (default: True)
        execute_func (Callable): The function to execute the agent.
        async_execute_func (Callable, Optional): Async version of the function. If not provided,
            an async wrapper will be automatically created around execute_func.
        llm_config (LLMConfig, optional): Configuration for the language model (minimal usage).
    """
    
    
    def __init__(
        self,
        name: str,
        description: str,
        inputs: List[dict],
        outputs: List[dict],
        execute_func: Callable,
        async_execute_func: Optional[Callable] = None,
        llm_config: Optional[LLMConfig] = None,
        **kwargs
    ):
        # Validate inputs
        if not callable(execute_func):
            raise ValueError("execute_func must be callable")
        
        if async_execute_func is not None and not callable(async_execute_func):
            raise ValueError("async_execute_func must be callable")
        
        # Store function references temporarily
        self._temp_execute_func = execute_func
        self._temp_async_execute_func = async_execute_func
        if llm_config is None:
            is_human = True
        else:
            is_human = False
        
        # Initialize parent with minimal configuration
        super().__init__(
            name=name,
            description=description,
            llm_config=llm_config,
            inputs=inputs,
            outputs=outputs,
            is_human=is_human,
            **kwargs
        )
        
        # Now that parent is initialized, set the actual attributes
        self.execute_func = execute_func
        self.async_execute_func = async_execute_func
        self.inputs = inputs
        self.outputs = outputs
    
    def validate_data(self, prompt: str, prompt_template, inputs: List[dict], outputs: List[dict], output_parser, parse_mode: str, parse_func: Callable, title_format: str):
        """Override to bypass prompt requirement for ActionAgent and validate inputs/outputs."""
        # For ActionAgent, we don't need prompts, so we skip the parent validation
        # We only validate our own requirements
        if not inputs:
            raise ValueError("`inputs` is required when creating an ActionAgent.")
        
        if not outputs:
            raise ValueError("`outputs` is required when creating an ActionAgent.")
        
        # Validate inputs structure
        for i, input_field in enumerate(inputs):
            if not isinstance(input_field, dict):
                raise ValueError(f"Input field {i} must be a dictionary, got {type(input_field)}")
            
            required_keys = ["name", "type", "description"]
            for key in required_keys:
                if key not in input_field:
                    raise ValueError(f"Input field {i} missing required key '{key}'")
            
            if not isinstance(input_field["name"], str):
                raise ValueError(f"Input field {i} 'name' must be a string, got {type(input_field['name'])}")
            
            if not isinstance(input_field["type"], str):
                raise ValueError(f"Input field {i} 'type' must be a string, got {type(input_field['type'])}")
            
            if not isinstance(input_field["description"], str):
                raise ValueError(f"Input field {i} 'description' must be a string, got {type(input_field['description'])}")
            
            # Check for duplicate input names
            input_names = [field["name"] for field in inputs]
            if len(input_names) != len(set(input_names)):
                raise ValueError(f"Duplicate input names found: {[name for name in input_names if input_names.count(name) > 1]}")
        
        # Validate outputs structure
        for i, output_field in enumerate(outputs):
            if not isinstance(output_field, dict):
                raise ValueError(f"Output field {i} must be a dictionary, got {type(output_field)}")
            
            required_keys = ["name", "type", "description"]
            for key in required_keys:
                if key not in output_field:
                    raise ValueError(f"Output field {i} missing required key '{key}'")
            
            if not isinstance(output_field["name"], str):
                raise ValueError(f"Output field {i} 'name' must be a string, got {type(output_field['name'])}")
            
            if not isinstance(output_field["type"], str):
                raise ValueError(f"Output field {i} 'type' must be a string, got {type(output_field['type'])}")
            
            if not isinstance(output_field["description"], str):
                raise ValueError(f"Output field {i} 'description' must be a string, got {type(output_field['description'])}")
            
            # Check for duplicate output names
            output_names = [field["name"] for field in outputs]
            if len(output_names) != len(set(output_names)):
                raise ValueError(f"Duplicate output names found: {[name for name in output_names if output_names.count(name) > 1]}")
    
    def create_customize_action(
        self, 
        name: str, 
        desc: str, 
        prompt: str, 
        prompt_template, 
        inputs: List[dict], 
        outputs: List[dict], 
        parse_mode: str, 
        parse_func: Optional[Callable] = None,
        output_parser: Optional[ActionOutput] = None,
        title_format: Optional[str] = "## {title}",
        custom_output_format: Optional[str] = None,
        tools: Optional[List] = None,
        max_tool_calls: Optional[int] = 5
    ) -> Action:
        """Override to create function-based action instead of LLM-based action."""
        # Use temporary attributes if they exist, otherwise use stored attributes
        if hasattr(self, '_temp_execute_func'):
            execute_func = self._temp_execute_func
            async_execute_func = self._temp_async_execute_func
            inputs = inputs
            outputs = outputs
        else:
            execute_func = self.execute_func
            async_execute_func = self.async_execute_func
            inputs = inputs
            outputs = outputs
        
        return self._create_function_action_with_params(name, execute_func, async_execute_func, inputs, outputs)
    
    def _create_function_action_input_type(self, name: str, inputs: List[dict]) -> Type[ActionInput]:
        """Create ActionInput type from input specifications."""
        action_input_fields = {}
        for field in inputs:
            required = field.get("required", True)
            if required:
                action_input_fields[field["name"]] = (str, Field(description=field["description"]))
            else:
                action_input_fields[field["name"]] = (Optional[str], Field(default=None, description=field["description"]))
        
        action_input_type = create_model(
            self._get_unique_class_name(
                generate_dynamic_class_name(f"{name} action_input")
            ),
            **action_input_fields,
            __base__=ActionInput
        )
        return action_input_type
    
    def _create_function_action_output_type(self, name: str, outputs: List[dict]) -> Type[ActionOutput]:
        """Create ActionOutput type from output specifications."""
        action_output_fields = {}
        for field in outputs:
            required = field.get("required", True)
            if required:
                action_output_fields[field["name"]] = (Any, Field(description=field["description"]))
            else:
                action_output_fields[field["name"]] = (Optional[Any], Field(default=None, description=field["description"]))
        
        action_output_type = create_model(
            self._get_unique_class_name(
                generate_dynamic_class_name(f"{name} action_output")
            ),
            **action_output_fields,
            __base__=ActionOutput
        )
        return action_output_type
    
    def _create_execute_method(self, execute_func: Callable):
        """Create the execute method for the action."""
        def execute_method(action_self, llm=None, inputs=None, sys_msg=None, return_prompt=False, **kwargs):
            # Validate inputs
            if inputs is None:
                inputs = {}
            
            # Validate that all required inputs are provided
            required_inputs = action_self.inputs_format.get_required_input_names()
            missing_inputs = [input_name for input_name in required_inputs if input_name not in inputs]
            if missing_inputs:
                raise ValueError(f"Missing required inputs: {missing_inputs}")
            
            # Validate input types (basic validation)
            for input_name, input_value in inputs.items():
                if input_name not in [field["name"] for field in self.inputs]:
                    logger.warning(f"Unexpected input '{input_name}' provided")
            
            # Execute function
            try:
                result = execute_func(**inputs)
            except Exception as e:
                # Create error output - try to use error field if it exists, otherwise use first available field
                try:
                    # Check if output format has an error field
                    output_fields = action_self.outputs_format.get_attrs()
                    if "error" in output_fields:
                        error_output = action_self.outputs_format(
                            error=f"Function execution failed: {str(e)}"
                        )
                    elif len(output_fields) > 0:
                        # Use the first field as error field
                        first_field = output_fields[0]
                        error_output = action_self.outputs_format(**{first_field: f"Error: {str(e)}"})
                    else:
                        # Fallback to creating a simple output with error message
                        error_output = action_self.outputs_format()
                except Exception as create_error:
                    # If all else fails, create a minimal output
                    logger.error(f"Failed to create error output: {create_error}")
                    error_output = action_self.outputs_format()
                return error_output, "Function execution"
            
            # Create success output using the parse method
            if isinstance(result, dict):
                # For dict results, use JSON parsing
                import json
                content = json.dumps(result, indent=2)
                parse_mode = "json"
            else:
                # For simple values, use string parsing
                content = str(result)
                parse_mode = "str"
            
            output = action_self.outputs_format.parse(
                content=content,
                parse_mode=parse_mode
            )
            return output, "Function execution"
        
        return execute_method
    
    def _create_async_execute_method(self, async_execute_func: Callable, execute_func: Callable):
        """Create the async execute method for the action."""
        async def async_execute_method(action_self, llm=None, inputs=None, sys_msg=None, return_prompt=False, **kwargs):
            # Validate inputs
            if inputs is None:
                inputs = {}
            
            # Validate that all required inputs are provided
            required_inputs = action_self.inputs_format.get_required_input_names()
            missing_inputs = [input_name for input_name in required_inputs if input_name not in inputs]
            if missing_inputs:
                raise ValueError(f"Missing required inputs: {missing_inputs}")
            
            # Validate input types (basic validation)
            for input_name, input_value in inputs.items():
                if input_name not in [field["name"] for field in self.inputs]:
                    logger.warning(f"Unexpected input '{input_name}' provided")
            
            # Execute async function
            try:
                if async_execute_func is not None:
                    result = await async_execute_func(**inputs)
                else:
                    # Use sync function in async context
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: execute_func(**inputs))
            except Exception as e:
                # Create error output - try to use error field if it exists, otherwise use first available field
                try:
                    # Check if output format has an error field
                    output_fields = action_self.outputs_format.get_attrs()
                    if "error" in output_fields:
                        error_output = action_self.outputs_format(
                            error=f"Async function execution failed: {str(e)}"
                        )
                    elif len(output_fields) > 0:
                        # Use the first field as error field
                        first_field = list(output_fields.keys())[0]
                        error_output = action_self.outputs_format(**{first_field: f"Error: {str(e)}"})
                    else:
                        # Fallback to creating a simple output with error message
                        error_output = action_self.outputs_format()
                except Exception as create_error:
                    # If all else fails, create a minimal output
                    logger.error(f"Failed to create error output: {create_error}")
                    error_output = action_self.outputs_format()
                return error_output, "Async function execution"
            
            # Create success output using the parse method
            if isinstance(result, dict):
                # For dict results, use JSON parsing
                import json
                content = json.dumps(result, indent=2)
                parse_mode = "json"
            else:
                # For simple values, use string parsing
                content = str(result)
                parse_mode = "str"
            
            output = action_self.outputs_format.parse(
                content=content,
                parse_mode=parse_mode
            )
            return output, "Async function execution"
        
        return async_execute_method
    
    def _create_function_action_with_params(self, name: str, execute_func: Callable, async_execute_func: Callable, inputs: List[dict], outputs: List[dict]) -> Action:
        """Create an action that executes the provided function with given parameters."""
        
        # Create input/output types
        action_input_type = self._create_function_action_input_type(name, inputs)
        action_output_type = self._create_function_action_output_type(name, outputs)
        
        # Create custom action class
        action_cls_name = self._get_unique_class_name(
            generate_dynamic_class_name(f"{name} function action")
        )
        
        # Create action class with function execution
        function_action_cls = create_model(
            action_cls_name,
            __base__=Action
        )
        
        # Create action instance
        function_action = function_action_cls(
            name=action_cls_name,
            description=f"Executes {execute_func.__name__} function",
            inputs_format=action_input_type,
            outputs_format=action_output_type
        )
        
        # Override execute methods - bind them properly to the action instance
        execute_method = self._create_execute_method(execute_func)
        async_execute_method = self._create_async_execute_method(async_execute_func, execute_func)
        
        # Bind the methods to the action instance
        function_action.execute = execute_method.__get__(function_action, type(function_action))
        function_action.async_execute = async_execute_method.__get__(function_action, type(function_action))
        
        return function_action
    
    def _create_function_action(self, name: str, execute_func: Callable, async_execute_func: Callable, inputs: List[dict], outputs: List[dict]) -> Action:
        """Create an action that executes the provided function."""
        return self._create_function_action_with_params(
            name,
            execute_func, 
            async_execute_func, 
            inputs, 
            outputs
        )
    
    def get_config(self) -> dict:
        """Override to include function information and handle None llm_config."""
        # Get base config but handle None llm_config
        config = self.get_customize_agent_info()
        
        # Handle llm_config safely
        if self.llm_config is not None:
            config["llm_config"] = self.llm_config.to_dict()
        else:
            config["llm_config"] = None
        
        # Add ActionAgent-specific information
        config.update({
            "execute_func_name": self.execute_func.__name__ if self.execute_func else None,
            "async_execute_func_name": self.async_execute_func.__name__ if self.async_execute_func else None,
            "inputs": self.inputs,
            "outputs": self.outputs
        })
        return config
    
    def save_module(self, path: str, ignore: List[str] = [], **kwargs) -> str:
        """Save the ActionAgent configuration to a JSON file.
        
        Args:
            path: File path where the configuration should be saved
            ignore: List of keys to exclude from the saved configuration
            **kwargs (Any): Additional parameters for the save operation
            
        Returns:
            The path where the configuration was saved
        """
        config = self.get_config()
        
        # Add ActionAgent-specific information
        config.update({
            "class_name": "ActionAgent",
            "execute_func_name": self.execute_func.__name__ if self.execute_func else None,
            "async_execute_func_name": self.async_execute_func.__name__ if self.async_execute_func else None,
            "inputs": self.inputs,
            "outputs": self.outputs
        })
        
        # Remove non-serializable items
        for ignore_key in ignore:
            config.pop(ignore_key, None)
        
        # Save to JSON file
        make_parent_folder(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        return path
    
    @classmethod
    def load_module(cls, path: str, llm_config: LLMConfig = None, **kwargs) -> "ActionAgent":
        """Load the ActionAgent from a JSON file.
        
        Args:
            path: The path of the file
            llm_config: The LLMConfig instance
            **kwargs: Additional keyword arguments
            
        Returns:
            ActionAgent: The loaded agent instance
            
        Raises:
            KeyError: If required functions are not found in the registry
        """
        # Load configuration
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract function names
        execute_func_name = config.get("execute_func_name")
        async_execute_func_name = config.get("async_execute_func_name")
        
        # Retrieve functions from registry
        execute_func = None
        async_execute_func = None
        
        if execute_func_name:
            if not ACTION_FUNCTION_REGISTRY.has_function(execute_func_name):
                raise KeyError(f"Function '{execute_func_name}' not found in registry. Please register it first.")
            execute_func = ACTION_FUNCTION_REGISTRY.get_function(execute_func_name)
        
        if async_execute_func_name:
            if not ACTION_FUNCTION_REGISTRY.has_function(async_execute_func_name):
                raise KeyError(f"Function '{async_execute_func_name}' not found in registry. Please register it first.")
            async_execute_func = ACTION_FUNCTION_REGISTRY.get_function(async_execute_func_name)
        
        # Create agent
        agent = cls(
            name=config["name"],
            description=config["description"],
            inputs=config["inputs"],
            outputs=config["outputs"],
            execute_func=execute_func,
            async_execute_func=async_execute_func,
            llm_config=llm_config,
            **kwargs
        )
        
        return agent