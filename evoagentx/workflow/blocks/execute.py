import dspy
from typing import Any
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Predictor, Reflector, Refiner, Test


class execute(block):
    def __init__(self, llm, n) -> None:        
        self.n = n
        self.predicotr = Predictor(llm=llm)
        self.Reflector = Reflector(llm=llm)
        self.Refiner = Refiner(llm=llm)
        self.Tester = Test(llm=llm)
        self.search_space = [0,1]

    def __call__(self,question, solution, benchmark):
        # executor + reflector
        
        predictor_solution = self.executor.execute(solution=solution)
        result = benchmark.evaluate(
            prediction = predictor_solution,
            label = benchmark.get_label(question)
        )


    def execute(self, question, solution, entry_point, benchmark):
        predictor_prediction = self.predicotr.execute(question=question)
        reflector_feedback = self.Reflector.execute(question=question, text=predictor_prediction['answer'])
        refiner_prediction = self.Refiner.execute(question=question, text=reflector_feedback['feedback'])
        tester_result = self.Tester.execute(problem=question, solution=refiner_prediction['answer'], entry_point=entry_point, benchmark=benchmark)
        return tester_result['result']
    
    def workflow_execute(self, question, solution, entry_point, benchmark):
        for i in range(self.n):
            predictor_prediction = self.predicotr.execute(question=question)
            reflector_feedback = self.Reflector.execute(question=question, text=predictor_prediction['answer'])

    
    def evaluate(self, question: dspy.Example, prediction: Any, *args, **kwargs):
        
        if isinstance(self.benchmark.get_train_data()[0], dspy.Example):
            # the data in original benchmark is a dspy.Example
            score = self.benchmark.evaluate(
                prediction=prediction, 
                label=self.benchmark.get_label(question)
            )
        elif isinstance(self.benchmark.get_train_data()[0], dict):
            # the data in original benchmark is a dict, convert the dspy.Example to a dict
            score = self.benchmark.evaluate(
                prediction=prediction, 
                label=self.benchmark.get_label(question.toDict()) # convert the dspy.Example to a dict
            )
        else:
            raise ValueError(f"Unsupported example type in `{type(self.benchmark)}`! Expected `dspy.Example` or `dict`, got {type(self.benchmark.get_train_data()[0])}")
        
        if isinstance(score, dict):
            score = self._extract_score_from_dict(score)
        
        return score