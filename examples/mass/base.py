from typing import Any, Dict
from evoagentx.workflow.operators import Operator, OperatorOutput
from evoagentx.models.base_model import BaseLLM, Field
from prompt import PREDICTOR_INIT_PROMPT
class PredictorOutput(OperatorOutput):
    answer: str = Field(default="", description="The predicted answer for the problem")

class Predictor(Operator):
    def __init__(self, llm: BaseLLM, **kwargs):
        name = "Predictor"
        description = "A predictor that takes a problem and outputs an answer"
        interface = "predictor(problem: str) -> dict with key 'answer' of type str"
        prompt = kwargs.pop("prompt",PREDICTOR_INIT_PROMPT)
        super().__init__(
            name=name,
            description=description,
            interface=interface,
            llm=llm,
            outputs_format=PredictorOutput,
            **kwargs
        )
    
    def execute(self, problem: str) -> Dict[str, Any]:
        """同步执行预测"""
        response = self.llm.generate(
            prompt=problem,
            parser=self.outputs_format,
            parse_mode="str"
        )
        return response.get_structured_data()
    
    async def async_execute(self, problem: str) -> Dict[str, Any]:
        """异步执行预测"""
        response = await self.llm.async_generate(
            prompt=problem,
            parser=self.outputs_format,
            parse_mode="str"
        )
        return response.get_structured_data()
    

    