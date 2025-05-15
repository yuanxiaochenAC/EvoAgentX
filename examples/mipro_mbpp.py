import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import MBPP  
from evoagentx.workflow import SequentialWorkFlowGraph, WorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info 
from evoagentx.optimizers import MiproOptimizer
from evoagentx.evaluators import MiproEvaluator
from evoagentx.utils.mipro_utils.settings import settings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class MBPPSplits(MBPP):
    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # randomly select 50 samples for dev and 100 samples for test
        self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]

def collate_func(example: dict) -> dict:
    return {"problem": example["prompt"]}

mbpp_graph_data = {
    "goal": "Generate a functional and correct Python code for the given problem.",
    "tasks": [
        {
            "name": "code_generate",
            "description": "Code generation for MBPP.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "code", "type": "str", "required": True, "description": "The generated code."}
            ],
            "prompt": "Generate a functional and correct Python code for the given problem.\n\nProblem: {problem}",
            "parse_mode": "str"
        }
    ] 
}

def main():
    # Initialize LLM
    llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    llm = OpenAILLM(config=llm_config)
    settings.lm = llm_config

    # Load benchmark
    benchmark = MBPPSplits()
    benchmark._load_data()
    trainset = benchmark._train_data
    testset = benchmark._test_data
    
    # Load workflow
    workflow_graph = SequentialWorkFlowGraph.from_dict(mbpp_graph_data)
    
    def evaluate_metric(example, prediction, trace=None):
        result = benchmark.evaluate(prediction, example)
        return result['pass@1']
    
    # Create MIPRO evaluator
    evaluate = MiproEvaluator(
        devset=testset,
        metric=evaluate_metric,
        num_threads=32,
        display_progress=True,
        display_table=False,
    )

    # Create MIPRO optimizer
    optimizer = MiproOptimizer(
        graph=workflow_graph,
        metric=evaluate_metric,
        prompt_model=llm,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_candidates=15,
        auto="medium",
        num_threads=32,
        log_dir="examples/mipro/output/logs",
    )

    # Run optimization
    with suppress_logger_info():
        best_program = optimizer.optimize(
            trainset=trainset,
            with_inputs={"problem": "prompt"},
        )
        
    output_path = r"C:\Users\31646\Desktop\EvoAgentX\examples\mipro\output\best_program_mbpp.json"
    best_program.save_module(output_path)
    
    # 保存评估结果
    with suppress_logger_info():
        post_results = evaluate(program=WorkFlowGraph.from_file(output_path), 
                              with_inputs={"problem": "prompt"})

if __name__ == "__main__":
    main()
