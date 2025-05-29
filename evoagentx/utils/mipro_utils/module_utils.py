import asyncio
from typing import Callable, Dict, Union, Awaitable, Type

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
    predictors : Dict[str, dspy.Predict]
        A mapping of task names to their corresponding DSPy predictors.
        Each predictor handles a specific task in the program.
    registry : ParamRegistry
        A registry that maintains the tunable parameters shared between
        predictors and the program.
    """

    def __init__(
        self,
        program: Union[Callable[..., dict], Callable[..., Awaitable[dict]]],
        predictors: Dict[str, dspy.Predict],
        registry: ParamRegistry,
    ):
        super().__init__()
        self.program = program
        self.predictors = predictors  # Mapping of task names to their predictors
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
        for name, predictor in self.predictors.items():
            signature = predictor.signature

            for field_name in signature.input_fields:
                # Prioritize values from config (may contain optimized values)
                value = predictor.config.get(field_name)
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
        if not isinstance(result, dict):
            raise ValueError("program() must return a dict.")

        return result
