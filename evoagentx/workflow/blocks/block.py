from evoagentx.models.base_model import BaseLLM

class block:
    def __init__(self) -> None:
        pass
    
    def execute(self, problem):

        raise NotImplementedError

    async def async_execute(self, problem):

        raise NotImplementedError