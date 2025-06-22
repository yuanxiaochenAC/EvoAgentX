import json
from typing import Any
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Debater, Predictor

class debate(block):
    def __init__(self,
                 predictor,
                 llm):
        self.n = 0
        self.debater = Debater(llm=llm)
        self.predictior = predictor
        self.search_space = [0,1,2,3,4]
        self.activate = True

    def __call__(self, problem, **kwargs) -> Any:
        context = kwargs.get('context', None)
        predictions = []
        for _ in range(2):
            prediction = self.predictior.execute(problem=problem, **kwargs)
            predictions.append(prediction['answer'])

        debater_prediction = self.debator.execute(problem=problem, solutions=predictions, context=context)
    
        return debater_prediction['answer'], {"problem": problem, "answer": debater_prediction['answer']}
        

    def execute(self, problem, solutions, **kwargs):
        context = kwargs.get('context', None)
        for i in range(self.n):
            prediction = self.debator.execute(problem=problem, solutions=solutions, context=context, **kwargs)
            index = prediction['index']
            solutions[int(index)] = prediction['answer']

        return prediction
    
    def save(self, path: str):
        params = {
            "debater": self.debator.prompt,
            "predictor": self.predictior.prompt
        }
        
        with open(path, "w") as f:
            json.dump(params, f)
    
    def load(self, path: str):
        with open(path, "r") as f:
            params = json.load(f)
            self.debater.prompt = params["debatr"]
            self.predictior.prompt = params["predictor"]
    
    def get_registry(self):
        return ["debater.debater.prompt"]