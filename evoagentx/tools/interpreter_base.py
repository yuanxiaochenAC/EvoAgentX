import abc

class BaseInterpreter(abc.ABC):
    @abc.abstractmethod
    def execute(self, code: str, codetype: str) -> str:
        """Checks the code for safety before execution."""
        pass
