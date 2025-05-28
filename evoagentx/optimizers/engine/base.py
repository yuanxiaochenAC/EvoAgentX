from typing import Any, Callable, Dict, List, Tuple, Optional
import abc
from .decorators import EntryPoint
from .registry import ParamRegistry

class BaseOptimizer(abc.ABC):
    def __init__(
        self,
        registry: ParamRegistry,
        program: Callable, 
        evaluator: Callable[[Dict[str, Any]], float],
        **kwargs
    ):
        """
        Abstract base class for optimization routines.

        Parameters:
        - registry (ParamRegistry): parameter access layer
        - evaluator (Callable): function that evaluates the result dict and returns a float
        - metric (str): name of the metric used for optimization (optional, for logging)
        - maximize (bool): whether to maximize (True) or minimize (False) the metric
        - max_trials (int): how many trials to run
        """
        self.registry = registry
        self.program = program
        self.evaluator = evaluator

    def get_param(self, name: str) -> Any:
        """Retrieve the current value of a parameter by name."""
        return self.registry.get(name)

    def set_param(self, name: str, value: Any):
        """Set the value of a parameter by name."""
        self.registry.set(name, value)

    def param_names(self) -> List[str]:
        """Return the list of all registered parameter names."""
        return self.registry.names()

    def apply_cfg(self, cfg: Dict[str, Any]):
        """Apply a configuration dictionary to the registered parameters."""
        for k, v in cfg.items():
            if k in self.registry.fields:
                self.registry.set(k, v)

    @abc.abstractmethod
    def optimize(self, program_entry: Optional[Callable] = None):
        """
        Abstract optimization loop. Should be implemented by subclasses.

        Parameters:
        - program_entry: callable that runs the program and returns output dict

        Returns:
        - (best_cfg, history): best config found and full search history
        """
        if program_entry is None:
            program_entry = EntryPoint.get_entry()
        if program_entry is None:
            raise RuntimeError("No entry function provided or registered.")
        print(f"Starting optimization from entry: {program_entry.__name__}")
        raise NotImplementedError