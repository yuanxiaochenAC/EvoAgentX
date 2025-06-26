import os 
import json
from dotenv import load_dotenv
from typing import Any

from evoagentx.benchmark import MATH
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.optimizers import MiproOptimizer
from evoagentx.utils.mipro_utils.register_utils import MiproRegistry
from evoagentx.workflow.operators import Predictor, Summarizer
from evoagentx.workflow.blocks.summarize import summarize
from evoagentx.workflow.blocks.aggregate import aggregate
from evoagentx.workflow.blocks.reflect import reflect
from evoagentx.workflow.blocks.debate import debate
from evoagentx.workflow.blocks.execute import execute
from evoagentx.optimizers.mass_optimizer import MassOptimiser
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_BOOTSTRAPPED_DEMOS = 1
MAX_LABELED_DEMOS = 0
AUTO = "light"
NUM_THREADS = 16
EVALUATION_ROUNDS = 1
class MathSplits(MATH):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # radnomly select 50 samples for training and 100 samples for test
        # self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._train_data = [full_test_data[idx] for idx in permutation[:10]]
        self._test_data = [full_test_data[idx] for idx in permutation[10:20]]

    # define the input keys. 
    # If defined, the corresponding input key and value will be passed to the __call__ method of the program, 
    # i.e., program.__call__(**{k: v for k, v in example.items() if k in self.get_input_keys()})
    # If not defined, the program will be executed with the entire input example, i.e., program.__call__(**example)
    def get_input_keys(self):
        return ["problem"]
    
    # the benchmark must have a `evaluate` method that receives the program's `prediction` (output from the program's __call__ method) 
    # and the `label` (obtained using the `self.get_label` method) and return a dictionary of metrics. 
    def evaluate(self, prediction: Any, label: Any) -> dict:
        return super().evaluate(prediction, label)

class WorkFlow():
    def __init__(self,
                 summarizer,
                 aggregater,
                 reflector,
                 debater,
                 executer) -> None:
        
        self.summarizer = summarizer
        self.aggregater = aggregater
        self.reflector = reflector
        self.debater = debater
        self.executer = executer
        self.blocks = [self.summarizer, self.aggregater, self.reflector, self.debater, self.executer]
    
    def __call__(self, problem: str, **kwargs):
        entry_point = kwargs.get("entry_point", None)
        context = kwargs.get("context", None)
        # Step 1: 获取总结的上下文
        context = self.summarizer.execute(problem, context = context)
        
        # Step 2: 生成候选解决方案
        if self.debater.n > 0:
            # 需要辩论，生成多个候选方案
            candidate_solutions = self.aggregater.execute(problem, context=context)
        else:
            # 没有辩论，使用self consistency
            candidate_solution, _ = self.aggregater(problem, context=context)
            candidate_solutions = [candidate_solution]
        
        # Step 3: 对每个候选方案进行反思优化
        solutions = []
        for solution in candidate_solutions:
            if self.reflector.n > 0:
                if self.executer.n > 0:
                    refined_solution = self.executer.execute(problem = problem, solution = solution, entry_point = kwargs.pop("entry_point", None), testcases = kwargs.pop("testcases", None))
                    solutions.append(refined_solution)
                else:
                    refined_solution = self.reflector.execute(problem = problem, solution = solution, context = context)
                    solutions.append(refined_solution)
            
        # Step 4: 通过辩论选择最佳答案
        final_answer = self.debater.execute(problem, solutions, context=context)
        
        return final_answer, {"problem":problem, "answer":final_answer}

    def save(self, path):
        params = {
            "summarizer": {
                "n": self.summarizer.n,
                "summarizer": self.summarizer.summarizer.prompt,
                "predictor": self.summarizer.predictor.prompt,
            },
            "aggregater":{
                "n": self.aggregater.n,
                "predictor": self.aggregater.predictor.prompt,
            },
            "reflector": {
                "n": self.reflector.n,
                "reflector": self.reflector.reflector.prompt,
                "refiner": self.reflector.refiner.prompt,
            },
            "debater": {
                "n": self.debater.n,
                "debater": self.debater.debater.prompt,
                "predictor": self.debater.predictor.prompt,
            },
            "executer": {
                "n": self.executer.n,
                "predictor": self.executer.predictor.prompt,
                "code_reflector": self.executer.code_reflector.prompt,
            }
        }
        with open(path, "w") as f:
            json.dump(params, f)

def get_save_path(program):
    return f"examples/mass/{program}"


def mipro_optimize(registry, block, optimizer_llm, save_path, benchmark):
    optimizer = MiproOptimizer(
        registry=registry,
        program=block,
        optimizer_llm=optimizer_llm,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MAX_LABELED_DEMOS,
        num_threads=NUM_THREADS,
        eval_rounds=EVALUATION_ROUNDS,
        auto=AUTO,
        save_path=save_path
    )
    
    optimizer.optimize(dataset=benchmark)
    return optimizer

def optimize_predictor(predictor, optimizer_llm, benchmark):
    registry = MiproRegistry()
    registry.track(predictor, "prompt", input_names=['problem', "context"], output_names=['answer'])
    
    optimizer = mipro_optimize(registry, predictor, optimizer_llm, get_save_path("predictor"), benchmark)
    return optimizer.evaluate(dataset = benchmark), optimizer.restore_best_program()

def optimize_summarizer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score):
    block = summarize(predictor=optimized_predictor, llm=executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "summarizer.prompt", input_names=['problem', 'context'], output_names=['summary', 'reasoning', 'answer'])
    registry.track(block, "predictor.prompt", input_names=['problem', 'context'], output_names=['summary', 'reasoning', 'answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/summarizer"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def optimize_aggregator(optimized_predictor, optimizer_llm, benchmark, predictor_score):
    block = aggregate(predictor=optimized_predictor)
    
    registry = MiproRegistry()
    registry.track(block, "predictor.prompt", input_names=['problem', 'context'], output_names=['answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/aggregator"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def optimize_reflector(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score):
    block = reflect(predictor=optimized_predictor, llm=executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "predictor.prompt", input_names=['problem', 'context'], output_names=['answer'])
    registry.track(block, "reflector.prompt", input_names=['problem', 'context'], output_names=['answer'])
    registry.track(block, "refiner.prompt", input_names=['problem', 'context'], output_names=['answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/reflector"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def optimize_debater(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score):
    block = debate(predictor=optimized_predictor, llm=executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "debater.prompt", input_names=['problem', 'context'], output_names=['answer'])
    registry.track(block, "predictor.prompt", input_names=['problem', 'context'], output_names=['answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/debater"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def optimize_executer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score):
    block = execute(predictor=optimized_predictor, benchamark = benchmark, llm = executor_llm)
    
    registry = MiproRegistry()
    registry.track(block, "predictor.prompt", input_names=['problem', 'entry_point', 'testcases'], output_names=['answer'])
    registry.track(block, "code_reflector.prompt", input_names=['problem', 'entry_point', 'testcases'], output_names=['answer'])
    
    optimizer = mipro_optimize(registry, block, optimizer_llm, get_save_path("mass/executer"), benchmark)
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    return optimized_block

def main():
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    optimizer_llm = OpenAILLM(config=optimizer_config)

    benchmark = MathSplits()
    
    # Step 0: 优化 Predictor
    predictor = Predictor(llm=executor_llm)
    
    # Test done
    # predictor_score, optimized_predictor = optimize_predictor(predictor, optimizer_llm, benchmark)

    optimized_predictor = Predictor(llm = executor_llm)

    predictor_score = 62.5

    # Step 1: 逐个优化每个block
    print("优化 summarizer...")
    optimized_summarizer = optimize_summarizer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 aggregator...")
    optimized_aggregator = optimize_aggregator(optimized_predictor, optimizer_llm, benchmark, predictor_score)
    
    print("优化 reflector...")
    optimized_reflector = optimize_reflector(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 debater...")
    optimized_debater = optimize_debater(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)
    
    print("优化 executer...")
    optimized_executer = optimize_executer(optimized_predictor, executor_llm, optimizer_llm, benchmark, predictor_score)

    # 构建最终工作流
    block_workflow = WorkFlow(
        summarizer=optimized_summarizer,
        aggregater=optimized_aggregator,
        reflector=optimized_reflector,
        debater=optimized_debater,
        executer=optimized_executer
    )

    mass = MassOptimiser(WorkFlow = block_workflow,
                         optimizer_llm = optimizer_llm,
                         max_labeled_demso = 0,
                         auto = "light",
                         save_path = "examples/mass/mass_optimization",
                         num_threads = 16)

    best_program = mass.optimize(benchmark = benchmark)

if __name__ == "__main__":
    main()