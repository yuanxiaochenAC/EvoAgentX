# TextGrad Optimizer Tutorial

This tutorial will guide you through the process of setting up and running the TextGrad optimizer in EvoAgentX. We'll use the GSM8K dataset as an example to demonstrate how to optimize the prompts
and system prompts in a multi-agent workflow.

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
from evoagentx.benchmark import GSM8K
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
Currently, `TextGradOptimizer` only supports `SequentialWorkFlowGraph`. See [Workflow Graph](../modules/workflow_graph.md) for more information on `SequentialWorkFlowGraph`. For this example, let us create a 
simple workflow to solve math problems. This workflow involves two
stages: a planning stage and a problem solving stage.

```python
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
        "system_prompt": "Your role is to make a plan to solve a math problem. Think about what steps are needed to solve the problem, what information is needed and how to calculate it. Do not attempt to solve the problem or perform any calculations. Your plan should be a list of steps with clear and concise instructions."
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
                "description": "the plan to solve a math problem"
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

workflow_graph = SequentialWorkFlowGraph(goal="Solve math problems", tasks=tasks)
```

### Step 2: Prepare the dataset

For this tutorial, we will use the GSM8K dataset which consists of 8.5K grade school math problems. The dataset is split into 7.5K training problems and 1K test problems. We will shuffle the training set and split it into 90% training set and 10% development set to track the optimization progress.

```python
class GSM8KSplits(GSM8K):
    def _load_data(self):
        super()._load_data()
        import numpy as np 
        np.random.seed(1)
        num_dev_samples = int(len(self._train_data) * 0.1)
        random_indices = np.random.permutation(len(self._train_data))
        self._dev_data = [self._train_data[i] for i in random_indices[:num_dev_samples]]
        self._train_data = [self._train_data[i] for i in random_indices[num_dev_samples:]]

gsm8k = GSM8KSplits()
```

We also need a collate function to convert the raw example to the expected input for the workflow.

```python
def collate_func(example: dict) -> dict:
    return {"problem": example["question"]}
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
    max_steps=4,
    eval_interval=2,
    eval_rounds=1,
    eval_config={"sample_k": 20},
    collate_func=collate_func,
    output_postprocess_func=None,
    max_workers=4,
    save_interval=1,
    save_path="./",
    rollback=True
)
```

### Running the Optimization

To start the optimization process:
```python
textgrad_optimizer.optimize(dataset=gsm8k)
```

After optimization, we can evaluate the workflow again to see the improvement.

```python
with suppress_logger_info():
    result = textgrad_optimizer.evaluate(dataset=gsm8k, eval_mode="test")
print(f"Evaluation result (after optimization):\n{result}")
```

The final graph at the end of the optimization is not necessarily the best graph. If you wish to restore the graph that performed best on the development set, simply call
```python
textgrad_optimizer.restore_best_graph()
```

`TextGradOptimizer` always saves the final workflow graph and the best workflow graph. It also saves graphs during optimization if `save_interval` is not `None`. You can also save the workflow graph manually by calling `textgrad_optimizer.save()`.


Note that `TextGradOptimizer` does not change the workflow structure but saving the workflow graph also saves the prompts and system prompts which will be different after optimization.
Below is an example of a saved workflow graph after one step of optimization using `TextGradOptimizer`.


```json
{
    "class_name": "SequentialWorkFlowGraph",
    "goal": "Solve math problems",
    "tasks": [
        {
            "name": "planning",
            "description": "make a plan to solve the math problem",
            "inputs": [
                {
                    "name": "problem",
                    "type": "string",
                    "description": "the math problem",
                    "required": true
                }
            ],
            "outputs": [
                {
                    "name": "plan",
                    "type": "string",
                    "description": "the plan to solve the math problem",
                    "required": true
                }
            ],
            "prompt": "Create a step-by-step plan to determine the solution to the following math problem related to a real-world scenario. Ensure each step is clear, concise, and logically connected. Use bullet points or numbered lists to present the steps. Highlight key information and consider using visual aids, such as diagrams, to enhance understanding. Include a step to verify the solution for accuracy and consider rounding where necessary. Maintain consistent terminology throughout the plan.\n\n<input>{problem}</input>",
            "system_prompt": "You are a PlanningAgent tasked with creating structured and effective plans to solve math problems. Your goal is to outline a clear sequence of steps that identifies necessary information and calculations without performing them. Ensure your plan is concise, logically organized, and highlights key information. Include methods for verifying the accuracy of the solution and consider using visual aids or simplified language to enhance understanding. Maintain consistency in terminology and presentation, and critically evaluate each step for its necessity and efficiency.",
            "parse_mode": "str",
            "parse_func": null,
            "parse_title": null
        },
        {
            "name": "problem_solving",
            "description": "solve the math problem",
            "inputs": [
                {
                    "name": "problem",
                    "type": "string",
                    "description": "the math problem",
                    "required": true
                },
                {
                    "name": "plan",
                    "type": "string",
                    "description": "the plan to solve the math problem",
                    "required": true
                }
            ],
            "outputs": [
                {
                    "name": "solution",
                    "type": "string",
                    "description": "the solution to the math problem",
                    "required": true
                }
            ],
            "prompt": "Task Description:\nYou are tasked with solving a math problem using a structured plan to ensure accuracy and clarity. \n\nProblem Overview:\n<input>{problem}</input>\n\nPlan Overview:\n<input>{plan}</input>\n\nInstructions:\n- Begin by summarizing the problem context and any assumptions.\n- Follow the plan step-by-step, ensuring each calculation is verified for accuracy.\n- Highlight key figures and results for clarity.\n- Conclude with a reflection on the results and suggest any potential improvements or insights.\n- Maintain consistent terminology and use clear mathematical notation throughout.",
            "system_prompt": "You are a ProblemSolvingAgent specializing in solving real-world math problems using structured plans. Your goal is to provide clear, concise, and logically structured solutions. Follow the given plan step-by-step, ensuring each step logically connects to the next. Use consistent and task-focused terminology, and provide brief explanations for your calculations to enhance understanding. Verify your final answer by checking the reasonableness of the result and include a brief justification. Maintain a professional tone, and aim for clarity and engagement in your responses.",
            "parse_mode": "str",
            "parse_func": null,
            "parse_title": null
        }
    ]
}
```

For a complete working example, please refer to [textgrad_optimizer.py](../../examples/textgrad_optimizer.py).