from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Summarizer, Predictor
class summarize(block):
    def __init__(
            self,
            llm,
            n,
    ):
        self.n = n
        self.summarizer = Summarizer(llm=llm)
        self.predictor = Predictor(llm=llm)
        self.search_space = [0,1,2,3,4]

    def execute(self, question, **kwargs):

        context = kwargs.pop('context', None)
        if not context:
            raise ValueError("Context is required for summarization. This dataset does not support summarization without context.")

        for i in range(self.n):
            summary = self.summarizer.execute(question=question, context=context)
            context = summary['summary']
        prediction = self.predictor.execute(question=question, context=context)
        
        return prediction['answer']
    
    async def async_execute(self, question, **kwargs):
        predictor_prediction = await self.predictor.async_execute(question=question, **kwargs)
        summary = await self.summarizer.async_execute(question=question, context=predictor_prediction['answer'])
        return summary['summary']

    def workflow_execute(self, question, **kwargs):
        context = kwargs.pop('context', None)
        if not context:
            raise ValueError("Context is required for summarization. This dataset does not support summarization without context.")

        for i in range(self.n):
            summary = self.summarizer.execute(question=question, context=context)
            context = summary['summary']
        
        return context