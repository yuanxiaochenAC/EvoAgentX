# TextGrad Optimizer Tutorial

This tutorial will guide you through the process of setting up and running the TextGrad optimizer in EvoAgentX. We'll use the [MATH](https://www.modelscope.cn/datasets/opencompass/competition_math) dataset as an example to demonstrate how to optimize the prompts
and system prompts in a workflow.

## 1. TextGrad
TextGrad uses textual feedback from LLM to improve text variables. In EvoAgentX, we use TextGrad to optimize 
agents' prompts and system prompts. For more information on TextGrad, see their [paper](https://arxiv.org/abs/2406.07496) and [GitHub](https://github.com/zou-group/textgrad).

## 2. TextGrad Optimizer
The TextGrad optimizer in EvoAgentX enables you to:

- Automatically optimize multi-agent workflows (prompts and/or system prompts)
- Evaluate optimization results on datasets

## 3. Setting Up the Environment

First, let's import the necessary modules for setting up the TextGrad optimizer:

```python
from evoagentx.benchmark import MATH
from evoagentx.optimizers import TextGradOptimizer
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info
```

### Configure the LLM Model
You'll need a valid API key to initialize the LLM. See [Quickstart](../quickstart.md) for more details on how to set up your API key.

`TextGradOptimizer` allows the use of different LLMs for workflow execution and optimization. For example, we can use GPT 4o-mini for workflow execution and GPT 4o for optimization.

```python
executor_config = OpenAILLMConfig(model="gpt-4o-mini")
executor_llm = OpenAILLM(config=executor_config)

optimizer_config = OpenAILLMConfig(model="gpt-4o")
optimizer_llm = OpenAILLM(config=optimizer_config)
```

## 3. Setting Up the Components

### Step 1: Initialize the Workflow
Currently, `TextGradOptimizer` only supports `SequentialWorkFlowGraph`. See [Workflow Graph](../modules/workflow_graph.md) for more information on `SequentialWorkFlowGraph`. For this example, let us create the
simplest workflow with only a single node.

```python
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

workflow_graph = SequentialWorkFlowGraph.from_dict(math_graph_data)
```

### Step 2: Prepare the dataset

For this tutorial, we will use the MATH dataset which consists of challenging competition mathematics problems,
spanning various difficulty levels and subject areas. The dataset is split into 7.5K training problems and 5K test problems. For demonstration purpose, let's take a smaller subset of the dataset to speed up the validation and evaluation process.

```python
class MathSplits(MATH):
    def _load_data(self):
        super()._load_data()
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # randomly select 50 samples for dev and 100 samples for test
        self._dev_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]

math_splits = MathSplits()
```

We also need a collate function to convert the raw example to the expected input for the workflow.

```python
def collate_func(example: dict) -> dict:
    return {"problem": example["problem"]}
```

During optimization, the `TextGradOptimizer` will evaluate the performance on the development set by default. Please make sure the dataset has a development set properly set up. You can either:
   - Use a dataset that already provides a development set
   - Split your dataset to create a development set (like in the example above)
   - Implement a custom dataset (inherits from `evoagentx.benchmark.Benchmark`) with development set support


## 4. Configuring and Running the TextGrad Optimizer

The TextGradOptimizer can be configured with various parameters to control the optimization process:

- `graph`: The workflow graph to optimize
- `optimize_mode`: The mode of optimization:
    * "all": optimize prompts and system prompts
    * "prompt": optimize only the prompts
    * "system_prompt": optimize only the system prompts
- `executor_llm`: The LLM used to execute the workflow
- `optimizer_llm`: The LLM used to optimize the workflow
- `batch_size`: The batch size for optimization
- `max_steps`: The maximum number of optimization steps
- `eval_interval`: The number of steps between evaluations
- `eval_rounds`: The number of evaluation rounds
- `eval_config`: The evaluation configuration during optimization (passed to `TextGradOptimizer.evaluate()`). For example, if we don't want to evaluate on the entire development set, we can set  `eval_config = {"sample_k": 100}` to only evaluate on 100 random samples from the development set.
- `collate_func`: The collate function to convert the raw example to the expected input for the workflow
- `output_postprocess_func`: The output postprocess function to process the output of the workflow for evaluation
- `max_workers`: The maximum number of workers for evaluation
- `save_interval`: The number of steps between saving the workflow graph
- `save_path`: The path to save the workflow graph
- `rollback`: Whether to rollback to the best workflow graph during optimization


```python
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
    output_postprocess_func=None,
    max_workers=20,
    save_interval=None,
    save_path="./",
    rollback=True
)
```

### Running the Optimization

To start the optimization process:
```python
textgrad_optimizer.optimize(dataset=math_splits)
```

After optimization, we can evaluate the workflow again to see the improvement.

```python
with suppress_logger_info():
    result = textgrad_optimizer.evaluate(dataset=math_splits, eval_mode="test")
print(f"Evaluation result (after optimization):\n{result}")
```

The final graph at the end of the optimization is not necessarily the best graph. If you wish to restore the graph that performed best on the development set, simply call
```python
textgrad_optimizer.restore_best_graph()
```

`TextGradOptimizer` always saves the final workflow graph and the best workflow graph. It also saves graphs during optimization if `save_interval` is not `None`. You can also save the workflow graph manually by calling `textgrad_optimizer.save()`.


Note that `TextGradOptimizer` does not change the workflow structure but saving the workflow graph also saves the prompts and system prompts which will be different after optimization.
Below is an example of a saved workflow graph after optimization using `TextGradOptimizer`.


```json
{
    "class_name": "SequentialWorkFlowGraph",
    "goal": "Answer the math question. The answer should be in box format, e.g., \\boxed{123}",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer generation for Math.",
            "inputs": [
                {
                    "name": "problem",
                    "type": "str",
                    "description": "The problem to solve.",
                    "required": true
                }
            ],
            "outputs": [
                {
                    "name": "answer",
                    "type": "str",
                    "description": "The generated answer.",
                    "required": true
                }
            ],
            "prompt": "Begin by assessing the complexity of the math problem to determine the appropriate level of detail required. For complex problems, provide a brief introduction to set the context and explain the relevance of key mathematical concepts. For simpler problems, focus on delivering a direct and concise solution.\n\nIdentify and apply relevant mathematical properties or theorems that can simplify the problem-solving process, such as the arithmetic sequence property. Prioritize methods that offer a concise and efficient solution, minimizing unnecessary steps while maintaining clarity.\n\nSolve the problem using the most direct and appropriate mathematical methodologies, ensuring each calculation step is accurate. Clearly explain the reasoning behind each step, enhancing understanding by providing brief explanations of why specific mathematical properties or methods are applicable.\n\nMaintain a smooth and coherent logical flow throughout the solution, using transitional phrases to connect different parts of the problem-solving process. Where applicable, compare alternative methods to solve the problem, discussing the benefits of each approach to provide a comprehensive understanding.\n\nEncourage the use of visual aids, such as diagrams or charts, to illustrate complex concepts and enhance comprehension when necessary. Explicitly state and verify any assumptions made during the problem-solving process, clarifying why certain methodologies are chosen.\n\nConclude with a verification step to confirm the solution's correctness, and present the final answer in a consistent format, such as \\boxed{{answer}}. Ensure that the final expression is in its simplest form and that all calculations are accurate and justified.\n\nProblem: <input>{problem}</input>",
            "system_prompt": "You are a highly intelligent assistant with expertise in mathematics. Your goal is to provide precise, clear, and logically structured solutions to mathematical problems. Begin each solution with a concise summary of the problem to set the context for the reader. Assess the complexity of the problem and adjust the level of detail accordingly, focusing on conciseness for simpler problems. Emphasize mathematical rigor by using appropriate tools such as inequalities and fractions, and ensure accuracy by double-checking each step of the simplification process. Provide detailed explanations for each step when necessary, clearly articulating the reasoning and purpose behind each mathematical operation. Use consistent mathematical notation and terminology throughout the solution, specifying when to use LaTeX-style notation for complex equations. Where applicable, incorporate visual aids such as diagrams to illustrate complex concepts. Conclude with a summary of key concepts and rules used, and include a verification step to confirm the accuracy of the final answer. Use a conversational tone and incorporate real-world analogies to make the explanation more relatable and engaging for the reader. Encourage critical thinking by comparing different problem-solving methods and highlighting the most efficient approach.",
            "parse_mode": "str",
            "parse_func": null,
            "parse_title": null
        }
    ]
}
```

For a complete working example, please refer to [examples/textgrad/math_textgrad.py](../../examples/textgrad/math_textgrad.py). You can also find TextGrad optimization scripts for other datasets in [examples/textgrad](../../examples/textgrad).