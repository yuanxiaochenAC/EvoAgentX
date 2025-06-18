from typing import Any
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Debator, Predictor

class debate(block):
    def __init__(self,
                  llm):
        self.n = 0
        self.debator = Debator(llm=llm)
        self.Predictior = Predictor(llm=llm)
        self.search_space = [0,1,2,3,4]
        self.activate = True

    def __call__(self, question) -> Any:
        predictions = []
        for _ in range(2):
            prediction = self.Predictior.execute(question=question)
            predictions.append(prediction['answer'])

        debater_prediction = self.debator.execute(question=question, solutiosn=predictions)
    
        return debater_prediction['answer']      
        

    def execute(self, question, solutions, **kwargs):
        for i in range(self.n):
            prediction = self.debator.execute(question=question, solutions=solutions,  **kwargs)
            index = prediction['index']
            solutions[int(index)] = prediction['answer']

        return prediction
    
    def workflow_execute(self, question, solutions, **kwargs):
        return self.execute(question, solutions, **kwargs)