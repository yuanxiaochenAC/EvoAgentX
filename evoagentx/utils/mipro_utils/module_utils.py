import copy
import asyncio
from typing import Callable, Dict, Union, Awaitable
from pydantic import Field
import dspy
from ...optimizers.engine.registry import ParamRegistry  # Replace with your own path
from typing import List
import warnings
from ...prompts.template import PromptTemplate

class PromptTuningModule(dspy.Module):
    """
    A prompt tuning module that manages interactions between predictors,
    parameter registry, and program functions.
    
    This module coordinates prompt optimization through:
    1. Maintaining a set of predictors for different tasks
    2. Synchronizing optimized parameters back to the program
    3. Executing the program with updated parameters
    
    Parameters
    ----------
    program : Union[Callable[..., dict], Callable[..., Awaitable[dict]]]
        The main program function to execute. Can be either synchronous or asynchronous.
        Must return a dictionary containing execution results.
    signature_dict : Dict[str, dspy.Signature]
        A mapping of task names to their corresponding DSPy signatures.
        Each signature defines the input/output structure for a specific task.
    registry : ParamRegistry
        A registry that maintains tunable parameters shared between
        predictors and the program.
    """

    @classmethod
    def from_registry(
        cls,
        program: Union[Callable[..., dict], Callable[..., Awaitable[dict]]],
        registry: ParamRegistry,
    ) -> "PromptTuningModule":
        """
        Factory method to create a PromptTuningModule from a registry and program.
        
        This method:
        1. Creates signatures for each field in the registry
        2. Initializes a PromptTuningModule with the program and signatures
        3. Sets up predictors for each signature
        
        Parameters
        ----------
        program : Union[Callable[..., dict], Callable[..., Awaitable[dict]]]
            The main program function to execute
        registry : ParamRegistry
            Registry containing tunable parameters
            
        Returns
        -------
        PromptTuningModule
            A configured PromptTuningModule instance
            
        Examples
        --------
        >>> registry = ParamRegistry()
        >>> registry.register("task1", "What is {topic}?")
        >>> registry.register("task2", PromptTemplate(system="You are helpful.", user="{query}"))
        >>> def my_program(**kwargs) -> dict:
        ...     return {"result": "done"}
        >>> module = PromptTuningModule.from_registry(my_program, registry)
        """
        from .signature_utils import signature_from_registry

        # Create signatures for each field in the registry
        signature_dict, signature_name2register_name = signature_from_registry(
            registry=registry,
        )
        
        # Create and return the module instance
        return cls(program=program, signature_dict=signature_dict, registry=registry, signature_name2register_name=signature_name2register_name)

    def __init__(
        self,
        program: Union[Callable[..., dict], Callable[..., Awaitable[dict]]],
        signature_dict: Dict[str, dspy.Signature],
        registry: ParamRegistry,
        signature_name2register_name: Dict[str, str],
    ):
        """
        Initialize a PromptTuningModule instance.
        
        Parameters
        ----------
        program : Union[Callable[..., dict], Callable[..., Awaitable[dict]]]
            The main program function to execute
        signature_dict : Dict[str, dspy.Signature]
            Mapping of task names to signatures
        registry : ParamRegistry
            Parameter registry
        signature_name2register_name : Dict[str, str]
            Mapping of signature names to register names
        """
        super().__init__()
        self.program = program
        self.predicts = []

        seen = set()
        for name, signature in signature_dict.items():
            if name in seen:
                raise ValueError(f"Duplicate name {name} in signature_dict")
            seen.add(name)
            self.predicts.append(dspy.Predict(signature, name=name))
        self.registry = registry
        self.signature_name2register_name = signature_name2register_name

    def reset(self):
        """
        Reset the module to its initial state.
        """
        self.registry.reset()

        for predict in self.predicts:
            signature = predict.signature

            signature_name = signature.__name__
            register_name = self.signature_name2register_name[signature_name]

            register_element = self.registry.get(register_name)

            if isinstance(register_element, PromptTemplate):
                predict.signature.instructions = register_element.instruction
                predict.demos = register_element.demonstrations
            elif isinstance(register_element, str):
                predict.signature.instructions = register_element
                predict.demos = []
            else:
                warnings.warn(f"Unsupported register element type: {type(register_element)}")
                # raise ValueError(f"Unsupported register element type: {type(register_element)}")
        
        return self



    def escape_braces(self, text):
        """
        Escape all braces in the text.
        
        Parameters
        ----------
        text : str
            Text that needs escaping
            
        Returns
        -------
        str
            Escaped text
        """
        def helper(s, start=0):
            result = ''
            i = start
            while i < len(s):
                if s[i] == '{':
                    inner, new_i = helper(s, i + 1)
                    result += '{{' + inner + '}}'
                    i = new_i
                elif s[i] == '}':
                    return result, i + 1
                else:
                    result += s[i]
                    i += 1
            return result, i

        escaped, _ = helper(text)
        return escaped
    
    def _validate_prompt(self, prompt: str, input_names: List[str], verbose: bool = True) -> str:
        """
        Validate if the generated prompt is valid. Currently only checks if required inputs are wrapped in braces.
        
        Parameters
        ----------
        prompt : str
            The prompt to validate
        input_names : List[str]
            List of required input names
        verbose : bool, optional
            Whether to show detailed information, defaults to True
            
        Returns
        -------
        str
            Validated and potentially modified prompt
        """
        modified_messages = []
        required_inputs = input_names
        missing_required_inputs = [name for name in required_inputs if f"{{{name}}}" not in prompt]
        if missing_required_inputs:
            input_values = "\n\n".join([f"{name}: {{{name}}}" for name in missing_required_inputs])
            prompt += f"\n\nThe followings are some required input values: \n{input_values}"
            modified_messages.append(f"added missing inputs: {', '.join(missing_required_inputs)}")

        prompt = self.escape_braces(prompt)
        for name in input_names:
            prompt = prompt.replace(f"{{{{{name}}}}}", f"{{{name}}}")
        prompt = prompt.replace(r"{{{{", r"{{").replace(r"}}}}", r"}}")

        if verbose and modified_messages:
            warnings.warn("Prompt modified: " + " | ".join(modified_messages))
        return prompt
    
    def get_field_type(self, field: Field) -> str:
        """
        Get the type of the field.
        
        Parameters
        ----------
        field : Field
            The field to get type from
            
        Returns
        -------
        str
            The field type
        """
        return field.json_schema_extra.get('__dspy_field_type') if field.json_schema_extra.get('__dspy_field_type') else None

    def is_prompt_template(self, register_name: str) -> bool:
        """
        Check if the register name is a prompt template.
        
        Parameters
        ----------
        register_name : str
            The register name to check
            
        Returns
        -------
        bool
            Whether it is a prompt template
        """
        return self.registry.get(register_name) is not None and isinstance(self.registry.get(register_name), PromptTemplate)

    def get_demos(self, demos: list) -> List[dict]:
        result = [] 
        for demo in demos:
            if isinstance(demo, dspy.Example):
                demo = demo.toDict()
            result.append(demo)
        return result
    
    def sync_predict_inputs_to_program(self):
        """
        Synchronize current input values from all predictors back to the registry.
        
        This method ensures that any optimized parameters in the predictors' configurations
        are properly reflected in the registry, which in turn affects program execution.
        
        Synchronization process:
        1. Iterate through all predictors
        2. For each predictor, check its signature's input fields
        3. If a field has a value in the predictor's config, update the registry
        
        Note: Values in predictor configs take precedence as they may contain
        optimized values from recent tuning iterations.
        """
        for predict in self.predicts:
            signature = predict.signature
            instruction = signature.instructions
            demos = predict.demos

            input_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'input']

            signature_name = signature.__name__
            register_name = self.signature_name2register_name[signature_name]
 
            if self.is_prompt_template(register_name):
                prompt_template: PromptTemplate = self.registry.get(register_name)
                prompt_template.instruction = instruction
                prompt_template.demonstrations = self.get_demos(demos)
                self.registry.set(register_name, prompt_template)
            else:
                instruction = self._validate_prompt(instruction, input_names)
                self.registry.set(register_name, instruction)
    
    def constrcut_trace(self, execution_data: dict) -> dict:
        """
        Construct the trace of the execution.
        
        Parameters
        ----------
        execution_data : dict
            Execution data
            
        Returns
        -------
        dict
            Trace information
        """
        trace: List[dict] = []
        for predict in self.predicts:
            input_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'input']
            output_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'output']

            input_dict = {}
            output_dict = {}

            # Check if input_names and output_names exist in execution data
            for name in input_names:
                if name not in execution_data:
                    warnings.warn(f"Input {name} not found in execution data")
            for name in output_names:
                if name not in execution_data:
                    warnings.warn(f"Output {name} not found in execution data")

            # Add input_names and output_names from execution data to trace
            for name in input_names:
                if name in execution_data:
                    input_dict[name] = execution_data[name]
            for name in output_names:
                if name in execution_data:
                    output_dict[name] = execution_data[name]
            
            trace_tuple = (predict, input_dict, output_dict)
            trace.append(trace_tuple)
        return trace

    def forward(self, **kwargs) -> dict:
        """
        Execute the program with synchronized parameters and optional inputs.
        
        This method:
        1. Synchronizes optimized prompts back to the program via registry
        2. Executes the program (handles both sync and async functions)
        3. Validates and returns the program's output
        
        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments to pass to the program function
            
        Returns
        -------
        dict
            The program's execution results
            
        Raises
        ------
        ValueError
            If the program doesn't return a dictionary
        """
        # 1. Sync optimized prompts back to program
        self.sync_predict_inputs_to_program()

        # 2. Execute the program (handle both sync/async)
        if asyncio.iscoroutinefunction(self.program):
            output, execution_data = asyncio.run(self.program(**kwargs)) if kwargs else asyncio.run(self.program())
        else:
            output, execution_data = self.program(**kwargs) if kwargs else self.program()

        trace = self.constrcut_trace(execution_data)

        # Use context manager to set trace
        if dspy.settings.trace is not None:
            dspy_trace = dspy.settings.trace
            dspy_trace.extend(trace)

        return output
    
    def save(self, path, save_program=False):
        return super().save(path, save_program)
    
    def load(self, path):
        return super().load(path)

    def deepcopy(self):
        """
        Deep copy the module.
        
        This is a tweak to the default Python deepcopy that only deep copies `self.parameters()`,
        and for other attributes, we just do a shallow copy.
        
        Returns
        -------
        PromptTuningModule
            A deep copy of the module
        """
        try:
            # If the instance itself is copyable, we can just deep copy it
            return copy.deepcopy(self)
        except Exception:
            pass

        # Create an empty instance
        new_instance = self.__class__.__new__(self.__class__)
        # Set attributes of the copied instance
        for attr, value in self.__dict__.items():
            if isinstance(value, dspy.Module):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    # Try to deep copy the attribute
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    try:
                        # Fall back to shallow copy if deep copy fails
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        # If even shallow copy fails, just copy the reference
                        setattr(new_instance, attr, value)
        
        return new_instance
        