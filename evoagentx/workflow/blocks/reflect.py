from typing import Any
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Predictor, Reflector, Refiner

class reflect(block):
    def __init__(
            self,
            llm,

    ):
        self.n = 0
        self.predictor = Predictor(llm=llm)
        self.reflector = Reflector(llm=llm)
        self.refiner = Refiner(llm=llm)
        self.search_space = [0, 1, 2, 3, 4]
        self.activate = True

    def __call__(self, question, **kwargs):
        """将 reflect 作为独立模块使用，返回最终精炼后的答案"""
        # 首先生成初始预测
        predictor_prediction = self.predictor.execute(question=question, **kwargs)
        answer = predictor_prediction['answer']
        
        reflector_prediction = self.reflector.execute(question=question, 
                                                        text=answer)
        refiner_prediction = self.refiner.execute(question=question, 
                                                    previous_answer=answer,
                                                    reflection=reflector_prediction['feedback'],
                                                    correctness=reflector_prediction['correctness'])
            
        answer = refiner_prediction['answer']
        
        return answer

    def execute(self, question, solution, **kwargs):
        """将 reflect 作为 workflow 组件使用，对给定的 solution 进行反思和精炼"""
        current_solution = solution
        
        for i in range(self.n):
            reflector_prediction = self.reflector.execute(question=question,
                                                          text=current_solution)
            
            refiner_prediction = self.refiner.execute(question=question,
                                                      previous_answer=current_solution,
                                                      reflection=reflector_prediction['feedback'],
                                                      correctness=reflector_prediction['correctness'])
            current_solution = refiner_prediction['answer']
        
        return current_solution