from evoagentx.optimizers.mipro_optimizer import MiproOptimizer
from evoagentx.models import OpenAILLMConfig, OpenAILLM, LiteLLM
from evoagentx.benchmark.humaneval import HumanEval
from evoagentx.utils.mipro_utils.settings import settings

# OPENAI_API_KEY = "OPENAI_API_KEY" 
import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    executor_llm = OpenAILLM(config=openai_config)
    humaneval = HumanEval()
    

    optimizer = MiproOptimizer(
        graph_path = "examples/output/saved_sequential_workflow.json",
        metric = print("hello world"),
        prompt_model = executor_llm,
        max_bootstrapped_demos = 4,
        max_labeled_demos = 4,
        auto = "medium",
        num_candidates = 10,
        num_threads = 10,
    )

    trainset = humaneval._test_data[:10]
    demos = optimizer._bootstrap_fewshot_examples(optimizer.graph, trainset, optimizer._set_random_seeds(9), None)

    print(demos)

if __name__ == "__main__":
    main()