import dspy
from typing import Callable, Awaitable, Type, Union
from optimizers.engine.registry import ParamRegistry
from dspy.signatures.signature import Signature
import asyncio


class PromptTuningModule(dspy.Module):
    def __init__(
        self,
        program: Union[Callable[..., dict], Callable[..., Awaitable[dict]]],
        signature: Type[Signature]
    ):
        super().__init__()
        self.program = program
        self.predict = dspy.Predict(signature)

    def forward(self, **kwargs) -> dict:

        if hasattr(self.program, "reset"):
            self.program.reset()

        # 执行 agent system，无论是否带输入
        if asyncio.iscoroutinefunction(self.program):
            result = asyncio.run(self.program(**kwargs)) if kwargs else asyncio.run(self.program())
        else:
            result = self.program(**kwargs) if kwargs else self.program()

        if not isinstance(result, dict):
            raise ValueError("program() must return a dict.")

        # 合并结果并调用 DSPy Predictor
        kwargs.update(result)
        return self.predict(**kwargs)