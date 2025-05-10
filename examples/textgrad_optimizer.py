from evoagentx.benchmark import GSM8K
from evoagentx.optimizers import TextGradOptimizer
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info
from dotenv import load_dotenv


tasks = [
    {
        "name": "planning",
        "description": "make a plan to solve the math problem",
        "inputs": [
            {
                "name": "problem",
                "type": "string",
                "required": True,
                "description": "the math problem"
            }
        ],
        "outputs": [
            {
                "name": "plan",
                "type": "string",
                "required": True,
                "description": "the plan to solve the math problem"
            }
        ],
        "prompt": "Make a plan to solve the following math problem:\n\n<input>{problem}</input>\n\n",
        "system_prompt": "Think about what steps are needed to solve the problem, what information is needed and how to calculate it. Do not attempt to solve the problem or perform any calculations. Your plan should be a list of steps with clear and concise instructions."
    },
    {
        "name": "problem_solving",
        "description": "solve the math problem",
        "inputs": [
            {
                "name": "problem",
                "type": "string",
                "required": True,
                "description": "the math problem"
            },
            {
                "name": "plan",
                "type": "string",
                "required": True,
                "description": "the plan to solve the math problem"
            }
        ],
        "outputs": [
            {
                "name": "solution",
                "type": "string",
                "required": True,
                "description": "the solution to the math problem"
            }
        ],
        "prompt": "The math problem:\n\n<input>{problem}</input>\n\nThe plan:\n<input>{plan}</input>",
        "system_prompt": "You will be given a math problem and a plan to solve the problem. Follow the plan to solve the problem. Check your calculation at every step."
    },
]


class GSM8KSplits(GSM8K):
    def _load_data(self):
        super()._load_data()
        # split the data into train and dev
        import numpy as np 
        np.random.seed(6)
        num_dev_samples = int(len(self._train_data) * 0.1)
        random_indices = np.random.permutation(len(self._train_data))
        self._dev_data = [self._train_data[i] for i in random_indices[:num_dev_samples]]
        self._train_data = [self._train_data[i] for i in random_indices[num_dev_samples:]]


def collate_func(example: dict) -> dict:
    return {"problem": example["question"]}


def main():
    load_dotenv()
    executor_config = OpenAILLMConfig(model="gpt-4o-mini")
    optimizer_config = OpenAILLMConfig(model="gpt-4o")
    executor_llm = OpenAILLM(config=executor_config)
    optimizer_llm = OpenAILLM(config=optimizer_config)
    gsm8k = GSM8KSplits()

    workflow_graph = SequentialWorkFlowGraph(goal="Solve math problems", tasks=tasks)

    # During optimization, periodically evaluate on the dev set to check the optimization progress.
    # This allows us to find the best prompts and rollback to it if the metrics get worse.
    # For demonstration purpose, we only evaluate on 10 random samples from the dev set.
    eval_config = {"eval_mode": "dev", "sample_k": 10}

    textgrad_optimizer = TextGradOptimizer(
        graph=workflow_graph, 
        optimize_mode="all",
        executor_llm=executor_llm, 
        optimizer_llm=optimizer_llm,
        batch_size=3,
        max_steps=4,
        eval_interval=2,
        eval_rounds=1,
        eval_config=eval_config,
        collate_func=collate_func,
        output_postprocess_func=None,
        max_workers=4,
        save_interval=1,
        save_path="./",
        rollback=True
    )

    # Before we start our optimization, let's evaluate the performance on the test set to get a baseline.
    # For demonstration purpose, we only evaluate on 20 samples to save time.
    test_indices = [x for x in range(20)]

    with suppress_logger_info():
        result = textgrad_optimizer.evaluate(dataset=gsm8k, eval_mode="test", indices=test_indices)
    print(f"Evaluation result (before optimization):\n{result}")

    # Start optimization
    textgrad_optimizer.optimize(gsm8k)

    # Evaluate the workflow again after optimization
    with suppress_logger_info():
        result = textgrad_optimizer.evaluate(dataset=gsm8k, eval_mode="test", indices=test_indices)
    print(f"Evaluation result (after optimization):\n{result}")


if __name__ == "__main__":
    main()
    