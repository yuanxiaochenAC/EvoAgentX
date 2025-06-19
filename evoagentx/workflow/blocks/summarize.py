from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Summarizer, Predictor

class summarize(block):
    def __init__(
            self,
            llm,
    ):
        self.n = 0
        self.summarizer = Summarizer(llm=llm)
        self.predictor = Predictor(llm=llm)
        self.search_space = [0, 1, 2, 3, 4]
        self.activate = True

    def __call__(self, question, **kwargs):
        """将 summarize 作为独立模块使用，返回最终的预测答案"""
        context = kwargs.pop('context', None)
        if not context:
            raise ValueError("Context is required for summarization. This dataset does not support summarization without context.")

        # 进行 n 轮总结，逐步精炼上下文
        summary = self.summarizer.execute(question=question, context=context)
        current_context = summary['summary']
        
        # 基于最终总结的上下文进行预测
        prediction = self.predictor.execute(question=question, context=current_context)
        
        return prediction['answer']

    def execute(self, question, **kwargs):
        """将 summarize 作为 workflow 组件使用，返回总结后的上下文"""
        context = kwargs.pop('context', None)
        if not context:
            raise ValueError("Context is required for summarization. This dataset does not support summarization without context.")

        # 进行 n 轮总结
        current_context = context
        for i in range(self.n):
            summary = self.summarizer.execute(question=question, context=current_context)
            current_context = summary['summary']
        
        return current_context