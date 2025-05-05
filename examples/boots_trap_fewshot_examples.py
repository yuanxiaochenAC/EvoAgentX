from evoagentx.optimizers.mipro_optimizer import MiproOptimizer
from evoagentx.models import OpenAILLMConfig, OpenAILLM, LiteLLM
from evoagentx.benchmark.humaneval import HumanEval
from evoagentx.utils.mipro_utils.settings import settings
import json
# OPENAI_API_KEY = "OPENAI_API_KEY" 
import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    executor_llm = OpenAILLM(config=openai_config)
    settings.lm = openai_config
    humaneval = HumanEval()
    
    print(settings.lm.temperature)

    def evaluate_metric(example, prediction, trace):
        return True

    optimizer = MiproOptimizer(
        graph_path = "examples/output/saved_sequential_workflow.json",
        metric = evaluate_metric,
        prompt_model = executor_llm,
        max_bootstrapped_demos = 4,
        max_labeled_demos = 4,
        auto = "light",
        num_candidates = 10,
        num_threads = 10,
    )

    trainset = humaneval._test_data[:100]
    # 设置完整的输出路径
    output_path = r"C:\Users\31646\Desktop\EvoAgentX\examples\mipro\output\demo_candidates.json"

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    demos = optimizer._bootstrap_fewshot_examples(optimizer.graph, trainset, ["prompt"], optimizer._set_random_seeds(9), None)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(demos, f, indent=2, ensure_ascii=False, default=str)

if __name__ == "__main__":
    main()