import copy
import asyncio
from typing import Callable, Dict, Union, Awaitable, Type, Optional

import dspy
from dspy import Signature
from ...optimizers.engine.registry import ParamRegistry  # 替换成你自己的路径


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

    @classmethod
    def from_registry(
        cls,
        program: Union[Callable[..., dict], Callable[..., Awaitable[dict]]],
        registry: ParamRegistry,
        output_field_name: str = "output",
        output_field_desc: str = "The final output.",
        output_field_type: Type = str,
        custom_output: Optional[Dict[str, str]] = None,
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
        output_field_name : str, default="output"
            Name for the output field in generated signatures
        output_field_desc : str, default="The final output."
            Default description for the output field
        output_field_type : Type, default=str
            Type annotation for the output field
        custom_output : Optional[Dict[str, str]], default=None
            Optional mapping of field names to custom output field descriptions

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
        signature_dict = signature_from_registry(
            registry=registry,
            output_field_name=output_field_name,
            output_field_desc=output_field_desc,
            output_field_type=output_field_type,
            custom_output=custom_output,
        )

        # Create and return the module
        return cls(program=program, signature_dict=signature_dict, registry=registry)

    def __init__(
        self,
        program: Union[Callable[..., dict], Callable[..., Awaitable[dict]]],
        signature_dict: Dict[str, dspy.Signature],
        registry: ParamRegistry,
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

            for field_name in signature.input_fields:
                # Prioritize values from config (may contain optimized values)
                value = predict.config.get(field_name)
                if value is not None:
                    self.registry.set(field_name, value)

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
            result = asyncio.run(self.program(**kwargs)) if kwargs else asyncio.run(self.program())
        else:
            result = self.program(**kwargs) if kwargs else self.program()

        # 3. Validate return type
        # if not isinstance(result, dict):
        #     raise ValueError("program() must return a dict.")

        return result

    def __deepcopy__(self, memo):
        
        # Create a new instance of the class
        new_instance = self.__class__.__new__(self.__class__)
        # Add to memo to prevent infinite recursion
        memo[id(self)] = new_instance
        
        # Deep copy all attributes except program
        new_instance.program = self.program  # Keep original program reference
        new_instance.predicts = copy.deepcopy(self.predicts, memo)
        new_instance.registry = copy.deepcopy(self.registry, memo)
        
        return new_instance
        