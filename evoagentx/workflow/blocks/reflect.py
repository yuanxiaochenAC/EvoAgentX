from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Predictor, Reflector, Refiner
class reflect(block):
    def __init__(
            self,
            llm,
            n,
    ):
        self.n = n
        self.predictor = Predictor(llm=llm)
        self.reflector = Reflector(llm=llm)
        self.refiner = Refiner(llm=llm)
        self.search_space = [0,1,2,3,4]

    def execute(self, question, **kwargs):
        
        predictor_prediction = self.predictor.execute(question=question, **kwargs)
        
        answer = predictor_prediction['answer']
        for i in range(self.n):
            reflector_prediction = self.reflector.execute(question=question, 
                                                          text = answer)
            refiner_prediction = self.refiner.execute(question=question, 
                                                      previous_answer = answer,
                                                      reflection = reflector_prediction['feedback'],
                                                      correctness = reflector_prediction['correctness'])
            
            answer = refiner_prediction['answer']
        return answer

    async def async_execute(self, question, **kwargs):
        predictor_prediction = await self.predictor.async_execute(question=question, **kwargs)
        
        answer = predictor_prediction['answer']
        for i in range(self.n):
            reflector_prediction = await self.reflector.async_execute(question=question, 
                                                                    text=answer)
            refiner_prediction = await self.refiner.async_execute(question=question, 
                                                                previous_answer=answer,
                                                                reflection=reflector_prediction['feedback'],
                                                                correctness=reflector_prediction['correctness'])
            
            answer = refiner_prediction['answer']
        return answer
    
    def workflow_execute(self, question, solution, **kwargs):
        for i in range(self.n):
            reflector_prediction = self.reflector.execute(qeustion = question,
                                                          text = solution)
            
            refiner_prediction = self.refiner.execute(question = question,
                                                      previous_answer = solution,
                                                      reflection = reflector_prediction['feedback'],
                                                      correctness = reflector_prediction['correctness'])
            solution = refiner_prediction['answer']
        return solution
    

    