import json
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Summarizer

class summarize(block):
    def __init__(
            self,
            predictor,
            llm,
    ):
        self.n = 0
        self.summarizer = Summarizer(llm=llm)
        self.predictor = predictor
        self.search_space = [0, 1, 2, 3, 4]
        self.activate = True

    def __call__(self, problem, **kwargs):
        """将 summarize 作为独立模块使用，返回最终的预测答案"""
        context = kwargs.pop('context', None)

        # 进行 n 轮总结，逐步精炼上下文
        summary = self.summarizer.execute(problem=problem, context=context)
        current_context = summary['summary']
        
        # 基于最终总结的上下文进行预测
        prediction = self.predictor.execute(problem=problem, context=current_context)
        
        return prediction['answer'], {"problem": problem, "summary":summary['summary'], "reasoning":prediction['reasoning'], "answer":prediction['answer']}

    def execute(self, question, **kwargs):
        """将 summarize 作为 workflow 组件使用，返回总结后的上下文"""
        context = kwargs.pop('context', None)
        if not context:
            raise ValueError("Context is required for summarization. This dataset does not support summarization without context.")

        # 进行 n 轮总结
        current_context = context
        for i in range(self.n):
            summary = self.summarizer.execute(problem=question, context=current_context)
            current_context = summary['summary']
        
        return current_context

    def save(self, path: str):
        params = {"summarizer": self.summarizer.prompt, "predictor": self.predictor.prompt}

        with open(path, "w") as f:
            json.dump(params, f)
    
    def load(self, path: str):
        with open(path, "r") as f:
            params = json.load(f)
            self.summarizer.prompt = params["summarizer"]
            self.predictor.prompt = params["predictor"]
    
    def get_registry(self):
        return ["summarizer.summarizer.prompt"]