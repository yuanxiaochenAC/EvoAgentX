import copy
import asyncio
from typing import Callable, Dict, Union, Awaitable
from pydantic import Field
import dspy
from ...optimizers.engine.registry import ParamRegistry  # 替换成你自己的路径
from typing import List
import warnings
from ...prompts.template import PromptTemplate

class PromptTuningModule(dspy.Module):
    """
    A DSPy module for prompt tuning that manages the interaction between predictors,
    a parameter registry, and a program function.

    This module coordinates the optimization of prompts by:
    1. Maintaining a set of predictors for different tasks
    2. Synchronizing optimized parameters back to the program
    3. Executing the program with the updated parameters

    Parameters
    ----------
    program : Union[Callable[..., dict], Callable[..., Awaitable[dict]]]
        The main program function to execute. Can be either synchronous or asynchronous.
        Must return a dictionary containing the execution results.
    signature_dict : Dict[str, dspy.Signature]
        A mapping of task names to their corresponding DSPy signatures.
        Each signature defines the input/output structure for a specific task.
    registry : ParamRegistry
        A registry that maintains the tunable parameters shared between
        predictors and the program.
    """

    # signature_name2register_name: Dict[str, str] = PrivateAttr()

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
            Registry containing the tunable parameters

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
        
        # Create and return the module
        return cls(program=program, signature_dict=signature_dict, registry=registry, signature_name2register_name=signature_name2register_name)

    def __init__(
        self,
        program: Union[Callable[..., dict], Callable[..., Awaitable[dict]]],
        signature_dict: Dict[str, dspy.Signature],
        registry: ParamRegistry,
        signature_name2register_name: Dict[str, str],
    ):
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

    def escape_braces(self, text):
        """
        This function escapes all the braces in the text.
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
        Check if the generated prompt is valid. Currently only check is the required inputs are wrapped in brackets. 
        """
        # prompt = prompt.replace("\"\"\"", "")
        # required_inputs = [inp["name"] for inp in task_info["inputs"] if inp["required"]]
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
        """
        return field.json_schema_extra.get('__dspy_field_type') if field.json_schema_extra.get('__dspy_field_type') else None

    def is_prompt_template(self, register_name: str) -> bool:
        """
        Check if the register_name is a prompt template.
        """
        return self.registry.get(register_name) is not None and isinstance(self.registry.get(register_name), PromptTemplate)


    def sync_predict_inputs_to_program(self):
        """
        Synchronizes the current input values from all predictors back to the registry.
        
        This method ensures that any optimized parameters in the predictors' configurations
        are properly reflected in the registry, which in turn affects the program execution.
        
        The synchronization process:
        1. Iterates through all predictors
        2. For each predictor, checks its signature's input fields
        3. If a field has a value in the predictor's config, updates the registry
        
        Note: Values in predictor configs take precedence as they may contain
        optimized values from recent tuning iterations.
        """
        for predict in self.predicts:
            signature = predict.signature
            instruction = signature.instructions
            demos = predict.demos

            input_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'input']

            # register_name = signature.__pydantic_extra__["register_name"] if signature.__pydantic_extra__["register_name"] else None
            # register_name = getattr(signature, 'register_name', None)

            signature_name = signature.__name__
            register_name = self.signature_name2register_name[signature_name]
 
            # self.registry.set(register_name, instruction)
            if self.is_prompt_template(register_name):
                prompt_template: PromptTemplate = self.registry.get(register_name)
                prompt_template.instruction = instruction
                prompt_template.demonstrations = demos
                self.registry.set(register_name, prompt_template)
            else:
                instruction = self._validate_prompt(instruction, input_names)
                self.registry.set(register_name, instruction)
                # todo: add demos to the instruction
    
    def constrcut_trace(self, execution_data: dict) -> dict:
        """
        Construct the trace of the execution.
        """
        trace: List[dict] = []
        for predict in self.predicts:
            # signature = predict.signature
            # instruction = signature.instructions

            input_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'input']
            output_names = [name for name, field in predict.signature.fields.items() if self.get_field_type(field) == 'output']

            input_dict = {}
            output_dict = {}

            # 先去检查执行数据中是否存在input_names和output_names
            for name in input_names:
                if name not in execution_data:
                    # raise ValueError(f"Input {name} not found in execution data")
                    warnings.warn(f"Input {name} not found in execution data")
            for name in output_names:
                if name not in execution_data:
                    # raise ValueError(f"Output {name} not found in execution data")
                    warnings.warn(f"Output {name} not found in execution data")

            # 如果存在，则将执行数据中的input_names和output_names加入到trace中
            # 这里name是没有的怎么办？
            for name in input_names:
                if name in execution_data:
                    input_dict[name] = execution_data[name]
                # else:
                #     input_dict[name] = None
            for name in output_names:
                if name in execution_data:
                    output_dict[name] = execution_data[name]
                # else:
                #     output_dict[name] = None
            
            trace_tuple = (predict, input_dict, output_dict)
            trace.append(trace_tuple)
        return trace


    def forward(self, **kwargs) -> dict:
        """
        Executes the program with synchronized parameters and optional inputs.

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

    def deepcopy(self):
        """Deep copy the module.

        This is a tweak to the default python deepcopy that only deep copies `self.parameters()`, and for other
        attributes, we just do the shallow copy.
        """
        try:
            # If the instance itself is copyable, we can just deep copy it.
            # Otherwise we will have to create a new instance and copy over the attributes one by one.
            return copy.deepcopy(self)
        except Exception:
            pass

        # Create an empty instance.
        new_instance = self.__class__.__new__(self.__class__)
        # Set attribuetes of the copied instance.
        for attr, value in self.__dict__.items():
            if isinstance(value, dspy.Module):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    # Try to deep copy the attribute
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    try:
                        # Fallback to shallow copy if deep copy fails
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        # If even the shallow copy fails, we just copy over the reference.
                        setattr(new_instance, attr, value)
        
        return new_instance
        