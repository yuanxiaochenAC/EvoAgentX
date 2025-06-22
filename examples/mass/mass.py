import os 
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
        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._test_data = [full_test_data[idx] for idx in permutation[100:200]]

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
                 reflecter,
                 debater,
                 executer) -> None:
        
        self.summarizer = summarizer
        self.aggregater = aggregater
        self.reflecter = reflecter
        self.debater = debater
        self.executer = executer
        self.blocks = [self.summarizer, self.aggregater, self.reflecter, self.debater, self.executer]
        

def get_save_path(program):
    return f"examples/mass/{program}"

def mipro_optimize(registry, program, llm, save_path, dataset):
    optimizer = MiproOptimizer(
        registry = registry,
        program = program,
        optimizer_llm = llm,
        max_bootstrapped_demos= MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos = MAX_LABELED_DEMOS,
        num_threads = NUM_THREADS,
        eval_rounds= EVALUATION_ROUNDS,
        auto = AUTO,
        save_path = save_path
    )

    optimizer.optimize(dataset = dataset)

    return optimizer

def main():
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    optimizer_llm = OpenAILLM(config=optimizer_config)

    benchmark = MathSplits()
    # Step 0: Optimize Predictor 0
    predictor = Predictor(llm = executor_llm)
    predictor_registry = MiproRegistry()
    predictor_registry.track(predictor, "prompt", input_names=['problem', "context"], output_names=['answer'])
    optimized_predictor = mipro_optimize(registry = predictor_registry, 
                                         program = predictor, 
                                         llm = optimizer_llm, 
                                         save_path = get_save_path("predictor"),
                                         dataset = benchmark)
    
    predictor_score = optimized_predictor.evaluate(dataset = benchmark, eval_mode = "test")

    # Step 1: Optimize each block
    # Step 1.1: Optimize block {Summarize}
    summarizer = summarize(predictor = optimized_predictor, llm = executor_llm)
    summarizer_registry = MiproRegistry()
    summarizer_registry.track(summarizer, "summarizer.prompt", input_names=['problem', 'context'], output_names=['summary', 'reasoning', 'answer'])
    summarizer_registry.track(summarizer, "predictor.prompt", input_names=['problem', 'context'], output_names=['summary','reasoning', 'answer'])

    summarizer_optimizer = mipro_optimize(registry = summarizer_registry, 
                                          program = summarizer, 
                                          llm = optimizer_llm, 
                                          save_path = get_save_path("summarizer"),
                                          dataset = benchmark)

    summarizer_score = summarizer_optimizer.evaluate(dataset = benchmark, eval_mode = "test")
    summarizer_influence = summarizer_score / predictor_score

    optimized_summarizer = summarizer_optimizer.restore_best_program()
    optimized_summarizer.influence_score = summarizer_influence


    # Step 1.2: Optimize block {Aggregate}
    aggregater = aggregate(predictor = optimized_predictor)
    aggregater_registry = MiproRegistry()
    aggregater_registry.track(aggregater, "predictor.prompt", input_names=['problem', 'context'], output_names=['answer'])
    aggregater_optimizer = mipro_optimize(registry = aggregater_registry, 
                                          program = aggregater, 
                                          llm = optimizer_llm, 
                                          save_path = get_save_path("aggregater"),
                                          dataset = benchmark)
    aggregater_score = aggregater_optimizer.evaluate(dataset = benchmark, eval_mode = "test")
    aggregater_influence = aggregater_score / predictor_score

    optimized_aggregater = aggregater_optimizer.restore_best_program()
    optimized_aggregater.influence_score = aggregater_influence


    # Step 1.3: Optimize block {Reflect}
    reflector = reflect(predictor = optimized_predictor, llm = executor_llm)
    reflector_registry = MiproRegistry()
    reflector_registry.track(reflector, "predictor.prompt", input_names=['problem', 'context'], output_names=['answer'])
    reflector_registry.track(reflector, "reflector.prompt", input_names=['problem', 'context'], output_names=['answer'])
    reflector_registry.track(reflector, "refiner.prompt", input_names=['problem', 'context'], output_names=['answer'])
    reflector_optimizer = mipro_optimize(registry = reflector_registry,
                                         program = reflector,
                                         llm = optimizer_llm,
                                         save_path = get_save_path("reflector"),
                                         dataset = benchmark)
    reflector_score = reflector_optimizer.evaluate(dataset = benchmark, eval_mode = "test")
    reflector_influence = reflector_score / predictor_score

    optimized_reflector = reflector_optimizer.restore_best_program()
    optimized_reflector.influence_score = reflector_influence

    # Step 1.4: Optimize block {Debate}
    debator = debate(predictor = optimized_predictor, llm = executor_llm)
    debator_registry = MiproRegistry()
    debator_registry.track(debator, "debator.prompt", input_names=['problem', 'context'], output_names=['answer'])
    debator_registry.track(debator, "predictor.prompt", input_names=['problem', 'context'], output_names=['answer'])
    debator_optimizer = mipro_optimize(registry = debator_registry,
                                       program  = debator,
                                       llm = optimizer_llm, 
                                       save_path = get_save_path("debater"),
                                       dataset = benchmark)
    debator_score = debator_optimizer.evaluate(dataset = benchmark, eval_mode = "test")
    debator_influence = debator_score / predictor_score

    optimized_debator = debator_optimizer.restore_best_program()
    optimized_debator.influence_score = debator_influence

    block_workflow = WorkFlow(summarizer = optimized_summarizer,
                              aggregater = optimized_aggregater,
                              reflector = optimized_reflector,
                              debater = optimized_debator,
                              executer = optimized_predictor)
    
    


if __name__ == "__main__":
    main()