import json
import dspy
import sys
import traceback
from typing import Any
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Predictor, CodeReflector
from evoagentx.utils.aflow_utils.data_utils import test_case_2_test_function

class execute(block):
    def __init__(self, predictor, benchamark, llm) -> None:        
        self.n = 0
        self.benchmark = benchamark
        self.predictor = predictor
        self.code_reflector = CodeReflector(llm=llm)
        self.search_space = [0,1]

    def __call__(self,problem, entry_point, testcases, **kwargs):


        predictor_prediction = self.predictor.execute(problem = problem, **kwargs)

        traceback = self.exec_code(solution = predictor_prediction['answer'], entry_point = entry_point, testcases = testcases)

        code_reflector_prediction = self.code_reflector.execute(question = problem, previous_solution = predictor_prediction['answer'], traceback = traceback)

        return code_reflector_prediction['answer'], {"problem":problem, 
                                                     "entry_point":entry_point, 
                                                     "reasoning":code_reflector_prediction["reasoning"], 
                                                     "correctness":code_reflector_prediction["correctness"], 
                                                     "answer":code_reflector_prediction["answer"]}


    def execute(self, problem, solution, entry_point, testcases):
    
        for i in range(self.n):
            traceback = self.exec_code(solution = solution, entry_point = entry_point, testcases = testcases)
            code_reflector_prediction = self.code_reflector.execute(question = problem, previous_solution = solution, traceback = traceback)
            solution = code_reflector_prediction["answer"]

        return solution


    def exec_code(self, solution, entry_point, testcases):
        if entry_point is None or testcases is None:
            return "No test cases available to execute"
        
        fail_cases = []
        for test_case in testcases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                error_msg = f"Test case failed: {test_case}\nAssertion Error: {str(e)}\nTraceback: {tb_str}"
                fail_cases.append(error_msg)
            except Exception as e:
                return f"Code execution error: {str(e)}"
        
        if fail_cases:
            return f"Test failed - {len(fail_cases)} case(s) failed:\n" + "\n---\n".join(fail_cases)
        else:
            return "All tests passed successfully"
    
    def get_registry(self):
        return ['executer.code_reflector.prompt']