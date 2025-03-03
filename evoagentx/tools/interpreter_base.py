import abc
from typing import Any

class BaseInterpreter(abc.ABC):
    @abc.abstractmethod
    def execute(self, code: str, codetype: str) -> str:
        """Checks the code for safety before execution."""
        pass

    @abc.abstractmethod
    def _update_namespace(self, alias: str, module_name: Any) -> None:
        """Update namespace for running the code"""
        pass