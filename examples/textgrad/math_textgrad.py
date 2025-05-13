from dotenv import load_dotenv
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import MATH  
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.optimizers import TextGradOptimizer
from evoagentx.core.callbacks import suppress_logger_info 
from evoagentx.core.logging import logger


load_dotenv()

class MathSplits(MATH):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # randomly select 50 samples for dev and 100 samples for test
        self._dev_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]


def collate_func(example: dict) -> dict:
    return {"problem": example["problem"]}


math_graph_data = {
    "goal": r"Answer the math question. The answer should be in box format, e.g., \boxed{123}",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer generation for Math.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "answer", "type": "str", "required": True, "description": "The generated answer."}
            ],
            "prompt": "Answer the math question. The answer should be in box format, e.g., \\boxed{{123}}\n\nProblem: {problem}",
            "parse_mode": "str"
        }
    ] 
}


def main(): 

    executor_config = OpenAILLMConfig(model="gpt-4o-mini")
    executor_llm = OpenAILLM(config=executor_config)

    optimizer_config = OpenAILLMConfig(model="gpt-4o")
    optimizer_llm = OpenAILLM(config=optimizer_config)

    # load benchmark 
    benchmark = MathSplits()

    # load workflow 
    workflow_graph = SequentialWorkFlowGraph.from_dict(math_graph_data)

    textgrad_optimizer = TextGradOptimizer(
        graph=workflow_graph, 
        optimize_mode="all",
        executor_llm=executor_llm, 
        optimizer_llm=optimizer_llm,
        batch_size=3,
        max_steps=20,
        eval_interval=1,
        eval_rounds=1,
        collate_func=collate_func,
        max_workers=20,
        save_interval=None,
        save_path="./",
        rollback=True
    )

    logger.info("Evaluating workflow on test set...")
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode="test")
    logger.info(f"Evaluation metrics (before optimization): {results}")

    logger.info("Optimizing workflow...")
    textgrad_optimizer.optimize(benchmark)
    textgrad_optimizer.restore_best_graph()

    logger.info("Evaluating workflow on test set...")
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode="test")
    logger.info(f"Evaluation metrics (after optimization): {results}")


if __name__ == "__main__":
    main() 
