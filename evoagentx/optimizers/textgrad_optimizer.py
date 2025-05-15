import textgrad as tg
from textgrad import Variable
from textgrad.loss import MultiFieldEvaluation
from textgrad.optimizer import TextualGradientDescent
from textgrad.autograd import StringBasedFunction
from pydantic import Field, PositiveInt
import os
from tqdm import tqdm
from typing import List, Literal, Optional
from copy import deepcopy
import numpy as np

from ..core.logging import logger
from ..core.module import BaseModule
from ..core.callbacks import suppress_logger_info
from ..models.base_model import BaseLLM
from ..benchmark.benchmark import Benchmark, CodingBenchmark
from ..evaluators import Evaluator
from ..workflow.workflow_graph import SequentialWorkFlowGraph, WorkFlowNode
from ..agents import CustomizeAgent
from ..prompts.optimizers.textgrad_optimizer import GENERAL_LOSS_PROMPT, CODE_LOSS_PROMPT, OPTIMIZER_SYSTEM_PROMPT, OPTIMIZER_CONSTRAINTS, PERSONAL_FINANCE_ADVISOR_EXAMPLE, FITNESS_COACH_EXAMPLE


class CustomAgentCall:
    """A custom agent call with textgrad.Variable inputs and output."""
    def __init__(self, agent: CustomizeAgent):
        self.agent = agent
        self.last_outputs: dict[str, str] = None
    
    def __call__(
        self, 
        prompt_template: Variable, 
        system_prompt: Variable,
        **inputs: Variable
    ) -> Variable:

        action = self.agent.actions[0]
        input_names = action.inputs_format.get_attrs()

        agent_inputs = {}

        for key, input_variable in inputs.items():
            if key in input_names:
                agent_inputs[key] = input_variable.value
            else:
                parsed_inputs: dict[str, str] = {k:v for k,v in input_variable.parsed_outputs.items() if k in input_names}
                agent_inputs.update(parsed_inputs)

        with suppress_logger_info():
            outputs = self.agent.execute(action_name=action.name, action_input_data=agent_inputs).content

        self.last_outputs = outputs.to_dict()

        return outputs.content


class TextGradAgent:
    """An agent that takes textgrad.Variable inputs and returns a textgrad.Variable response.
    This class is used to replace EvoAgentX CustomizeAgent in SequentialWorkFlowGraph to allow TextGrad optimization.
    """
    def __init__(
        self, 
        agent: CustomizeAgent,
        optimize_mode: Literal["all", "system_prompt", "prompt_template"] = "all"
    ):
        self.name = agent.name

        require_grad = {
            "all": {"system_prompt": True, "prompt_template": True}, 
            "system_prompt": {"system_prompt": True, "prompt_template": False}, 
            "prompt_template": {"system_prompt": False, "prompt_template": True}
        }

        system_prompt_require_grad = require_grad[optimize_mode]["system_prompt"]
        prompt_template_require_grad = require_grad[optimize_mode]["prompt_template"]

        self.system_prompt = Variable(
            agent.system_prompt, 
            requires_grad=system_prompt_require_grad, 
            role_description=f"{self.name}'s system prompt"
        )
        
        self.prompt_template = Variable(
            agent.prompt, 
            requires_grad=prompt_template_require_grad, 
            role_description=f"{self.name}'s instruction prompt template"
        )

        self._agent_call = CustomAgentCall(agent)
        self.forward = StringBasedFunction(self._agent_call, agent.description)
        self.output_description = " and ".join(agent.actions[0].outputs_format.get_attr_descriptions().values())
        self.last_output = None

    def __call__(self, inputs: dict[str, Variable]) -> Variable:
        """Given textgrad.Variable inputs, generates a textgrad.Variable output."""

        forward_inputs = dict(
            {
                "prompt_template": self.prompt_template, 
                "system_prompt": self.system_prompt,
                **inputs
            }
        )

        output_variable = self.forward(forward_inputs, self.output_description)
        output_variable.parsed_outputs = self._agent_call.last_outputs
        self.last_output = output_variable
        return output_variable



class TextGradOptimizer(BaseModule):
    """Optimizes the prompt templates and system prompts in a workflow using TextGrad.
    For more information, see https://github.com/zou-group/textgrad.
    """
    graph: SequentialWorkFlowGraph = Field(description="The workflow to optimize.")
    optimize_mode: Literal["all", "system_prompt", "prompt_template"] = Field(default="all", description="The mode to optimize the workflow.")
    executor_llm: BaseLLM = Field(default=None, description="The LLM to use for execution.")
    optimizer_llm: BaseLLM = Field(default=None, description="The LLM to use for optimization.")
    batch_size: PositiveInt = Field(default=1, description="The batch size for optimization.")
    max_steps: PositiveInt = Field(default=10, description="The maximum number of optimization steps.")
    evaluator: Evaluator = Field(default=None, description="The evaluator to perform evaluation during optimization.")
    eval_interval: Optional[PositiveInt] = Field(default=None, description="Evaluates performance every `eval_interval` steps.")
    eval_rounds: PositiveInt = Field(default=1, description="The number of times to evaluate the performance.")
    eval_config: dict = Field(default={}, description="The configuration for evaluation. The keys are the arguments of `TextGradOptimizer.evaluate`.")
    save_interval: Optional[PositiveInt] = Field(default=None, description="Save the workflow every `save_interval` steps.")
    save_path: str = Field(default="./", description="The path to save the optimized workflow.")
    rollback: bool = Field(default=True, description="Whether to rollback to the best graph after each evaluation during optimization.")


    def init_module(self, **kwargs):
        self._snapshot: List[dict] = []
        self.output_lookup = self._create_output_lookup()
        

    def _init_textgrad(self, dataset: Benchmark):
        # Disable TextGrad's short variable value to allow the optimizer to receive the full variable value
        def disable_short_variable_value(self):
            return self.value
        Variable.get_short_value = disable_short_variable_value

        # Textgrad engine
        self.optimizer_engine = tg.get_engine(self.optimizer_llm.config.model)

        # Textgrad loss
        if isinstance(dataset, CodingBenchmark):
            loss_prompt = CODE_LOSS_PROMPT
            role_descriptions = ["code snippet to evaluate", "the task, the test result of the code snippet, and the correct code"]
        else:
            loss_prompt = GENERAL_LOSS_PROMPT
            role_descriptions = ["response to evaluate", "correct answer"]
        evaluation_instruction = Variable(loss_prompt, requires_grad=False, role_description="evaluation instruction")
        self.loss_fn = MultiFieldEvaluation(evaluation_instruction, role_descriptions, self.optimizer_engine)

        # Create textgrad agents
        self._create_textgrad_agents()
        
        # Textgrad optimizer
        if self.optimize_mode == "all":
            optimize_variables = self._get_all_system_prompts() + self._get_all_prompt_templates()
        elif self.optimize_mode == "system_prompt":
            optimize_variables = self._get_all_system_prompts()
        elif self.optimize_mode == "prompt_template":
            optimize_variables = self._get_all_prompt_templates()
        
        self.textgrad_optimizer = TextualGradientDescent(
            parameters=optimize_variables, 
            engine=self.optimizer_engine,
            constraints=OPTIMIZER_CONSTRAINTS,
            optimizer_system_prompt=OPTIMIZER_SYSTEM_PROMPT,
            in_context_examples=[PERSONAL_FINANCE_ADVISOR_EXAMPLE, FITNESS_COACH_EXAMPLE]
        )


    def optimize(self, dataset: Benchmark):
        """Optimizes self.graph using `dataset`."""
        self._init_textgrad(dataset)

        for step in tqdm(range(self.max_steps)):
            idx = [x + step*self.batch_size for x in range(self.batch_size)]
            batch = dataset.get_train_data(indices=idx)
            inputs = [self.evaluator.collate_func(x) for x in batch]
            labels = dataset.get_labels(batch)

            self.step(inputs, labels, dataset)

            if self.eval_interval is not None and (step + 1) % self.eval_interval == 0:
                logger.info(f"Evaluating the workflow at step {step+1} ...")
                with suppress_logger_info():
                    metrics = self.evaluate(dataset, **self.eval_config)
                self.log_snapshot(self.graph, metrics)
                logger.info(f"Step {step+1} metrics: {metrics}")

                # If rollback is enabled, keep track of the best snapshot
                if self.rollback:
                    if len(self._snapshot) == 1:
                        best_snapshot = self._snapshot[-1]
                        best_average_score = np.mean(list(metrics.values()))
                    else:
                        current_average_score = np.mean(list(metrics.values()))
                        
                        if current_average_score >= best_average_score:
                            # If the current average score is better than the best average score, update the best snapshot
                            best_snapshot = self._snapshot[-1]
                            best_average_score = current_average_score
                        else:
                            # If the current average score is worse than the best average score, roll back to the best snapshot
                            logger.info(f"Metrics are worse than the best snapshot which has {best_snapshot['metrics']}. Rolling back to the best snapshot.")
                            best_graph = SequentialWorkFlowGraph.from_dict(best_snapshot["graph"])
                            self.graph = best_graph
                            self._create_textgrad_agents()

            if self.save_interval is not None and (step + 1) % self.save_interval == 0:
                logger.info(f"Saving the workflow at step {step+1} ...")
                self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_step_{step+1}.json"))

        logger.info(f"Reached the maximum number of steps {self.max_steps}. Optimization has finished.")
        self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_final.json"))

        # Saves the best graph
        if len(self._snapshot) > 0:
            best_graph = self._select_graph_with_highest_score()
            self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_best.json"), graph=best_graph)

        
    def step(self, inputs: List[dict[str, str]], labels: List[str|dict[str, str]], dataset: Benchmark):
        """Performs one optimization step using a batch of data."""
        losses = []
        for input, label in zip(inputs, labels):
            output = self.forward(input)
            if isinstance(label, str):
                label = Variable(label, requires_grad=False, role_description="correct answer for the query")
            elif isinstance(label, dict):
                if not isinstance(dataset, CodingBenchmark):
                    raise ValueError("Label must be a string for non-coding benchmarks.")
                end_node_name = self.graph.find_end_nodes()[0]
                end_node = self.graph.get_node(end_node_name)
                output_name = end_node.outputs[0].name
                code = output.parsed_outputs[output_name]
                label = self._format_code_label(code, label, dataset)
                label = Variable(label, requires_grad=False, role_description="the task, the test result, and the correct code")

            loss = self.loss_fn([output, label])
            losses.append(loss)
            
        total_loss = tg.sum(losses)
        total_loss.backward(self.optimizer_engine)
        self.textgrad_optimizer.step()
        self.textgrad_optimizer.zero_grad()

        # Checks if all the prompt templates contain the required inputs.
        # If not, fix them by appending the input placeholders at the end.
        for node in self.graph.nodes:
            prompt_template = node.textgrad_agent.prompt_template.value
            prompt_template = self._add_missing_input_placeholder(prompt_template, node)
            node.textgrad_agent.prompt_template.value = prompt_template

        self._update_workflow_graph()


    def forward(self, inputs: dict[str, str]) -> Variable:
        """Returns the final output from the workflow."""
        self._visited_nodes = set()
        end_node = self.graph.find_end_nodes()[0]
        input_variables = self._initial_inputs_to_variables(inputs)
        output = self._compute_node(end_node, input_variables)
        return output


    def evaluate(
        self, 
        dataset: Benchmark, 
        eval_mode: str = "dev", 
        graph: Optional[SequentialWorkFlowGraph] = None,
        indices: Optional[List[int]] = None,
        sample_k: Optional[int] = None,
        **kwargs
    ) -> dict:
        """Evaluate the workflow. If `graph` is provided, use the provided graph for evaluation. Otherwise, use the graph in the optimizer. 
        
        Args:
            dataset (Benchmark): The dataset to evaluate the workflow on.
            eval_mode (str): The evaluation mode. Choices: ["test", "dev", "train"].
            graph (SequentialWorkFlowGraph, optional): The graph to evaluate. If not provided, use the graph in the optimizer.
            indices (List[int], optional): The indices of the data to evaluate the workflow on.
            sample_k (int, optional): The number of data to evaluate the workflow on. If provided, a random sample of size `sample_k` will be used.
        
        Returns:
            dict: The metrics of the workflow evaluation.
        """
        if graph is None:
            graph = self.graph

        metrics_list = []
        for i in range(self.eval_rounds):
            eval_info = [
                f"[{type(graph).__name__}]", 
                f"Evaluation round {i+1}/{self.eval_rounds}", 
                f"Mode: {eval_mode}"
            ]
            if indices is not None:
                eval_info.append(f"Indices: {len(indices)} samples")
            if sample_k is not None:
                eval_info.append(f"Sample size: {sample_k}")
            logger.info(" | ".join(eval_info))
            metrics = self.evaluator.evaluate(
                graph=graph, 
                benchmark=dataset, 
                eval_mode=eval_mode, 
                indices=indices, 
                sample_k=sample_k,
                update_agents=True, 
                **kwargs
            )
            metrics_list.append(metrics)
        avg_metrics = self.evaluator._calculate_average_score(metrics_list)
        
        return avg_metrics


    def save(self, path: str, graph: Optional[SequentialWorkFlowGraph] = None, ignore: List[str] = []):
        """Save the workflow graph containing the optimized prompts to a file. 

        Args:
            path (str): The path to save the workflow graph.
            graph (SequantialWorkFlowGraph, optional): The graph to save. If not provided, use the graph in the optimizer.
            ignore (List[str]): The keys to ignore when saving the workflow graph.
        """
        if graph is None:
            graph = self.graph
        graph.save_module(path, ignore=ignore)


    def log_snapshot(self, graph: SequentialWorkFlowGraph, metrics: dict):
        """Log the snapshot of the workflow."""
        self._snapshot.append(
            {
                "index": len(self._snapshot),
                "graph": deepcopy(graph.get_graph_info()),
                "metrics": metrics,
            }
        )


    def restore_best_graph(self):
        """Restore the best graph from the snapshot and set it to `self.graph`."""
        if len(self._snapshot) == 0:
            logger.info("No snapshot found. No graph to restore.")
            return

        best_graph, best_metrics = self._select_graph_with_highest_score(return_metrics=True)
        self.graph = best_graph
        logger.info(f"Restored the best graph from snapshot with metrics {best_metrics}")


    def _format_code_label(self, code: str, label:dict[str, str], dataset: CodingBenchmark) -> str:
        """Formats the label for coding tasks to include the task, the test result, and the correct code.

        Args:
            code: The code to evaluate.
            label: A dictionary with keys "task_id", "test", "entry_point", and "canonical_solution".
            dataset: A CodingBenchmark instance with `check_solution` method.
        
        Returns:
            The formatted label which includes the task, the test result, and the correct code.
        """

        task_id = label["task_id"]
        prompt = dataset.get_example_by_id(task_id)["prompt"]
        test = label["test"]
        entry_point = label["entry_point"]

        state, message = dataset.check_solution(
            task_id=task_id,
            solution=prompt + "\n" + code,
            test=test,
            entry_point=entry_point
        )

        if state != dataset.SUCCESS:
            message = message.replace("Solution", "Failed Code")

        formatted_label = f"## Task:\n{prompt}\n\n## Result on test:\n{message}\n\n## Correct Solution:\n{label['canonical_solution']}"
        return formatted_label


    def _initial_inputs_to_variables(self, initial_inputs: dict[str, str]) -> dict[str, Variable]:
        """Converts inputs to the initial nodes to textgrad variables."""
        variables = {}
        initial_nodes = self.graph.find_initial_nodes()
        for initial_node in initial_nodes:
            for key, value in initial_inputs.items():
                for input in self.graph.get_node(initial_node).inputs:
                    if input.name == key:
                        variable_description = input.description
                        break
                initial_input_variable = Variable(
                    value,
                    requires_grad=False,
                    role_description=variable_description,
                )
                variables[key] = initial_input_variable
        return variables


    def _compute_node(self, node: str | WorkFlowNode, initial_inputs: dict[str, Variable]) -> Variable:
        """Computes the output of a node in the workflow graph by recursively computing the required inputs.

        Args:
            node: The node to compute the output of.
            initial_inputs: The initial inputs to the workflow that are not from any node in the workflow (e.g., user query).

        Returns:
            The output of the node as a textgrad.Variable.
        """
        if isinstance(node, str):
            node = self.graph.get_node(node)

        if node.name in self._visited_nodes:
            return node.textgrad_agent.last_output

        input_variables: dict[str, Variable] = {}    # inputs to TextGradAgent
        input_node_names: set[str] = set()           # which nodes we need to compute the output of      

        for input in node.inputs:
            if input.name in initial_inputs:
                input_variables[input.name] = initial_inputs[input.name]
            else:
                input_node_names.add(self.output_lookup[input.name])
        
        # if the input is from another node, compute the output of that node
        for node_name in input_node_names:
            input_variables[node_name] = self._compute_node(node_name, initial_inputs)

        output_variable = node.textgrad_agent(input_variables)
        self._visited_nodes.add(node.name)
        return output_variable
        
   
    def _create_textgrad_agent(self, node: str | WorkFlowNode) -> TextGradAgent:
        """Creates a textgrad agent for a given node in a WorkFlowGraph."""
        if isinstance(node, str):
            node = self.graph.get_node(node)

        agent_dict = node.agents[0]
        agent = CustomizeAgent(llm=self.executor_llm, **agent_dict)

        textgrad_agent = TextGradAgent(agent, self.optimize_mode)
        return textgrad_agent


    def _create_textgrad_agents(self):
        """Creates textgrad agents for all nodes in the workflow graph."""
        for node in self.graph.nodes:
            node.textgrad_agent = self._create_textgrad_agent(node)


    def _update_workflow_graph(self):
        """Updates the workflow graph with the latest prompts from the textgrad optimization."""
        for node in self.graph.nodes:
            node.agents[0]['prompt'] = node.textgrad_agent.prompt_template.value
            node.agents[0]['system_prompt'] = node.textgrad_agent.system_prompt.value


    def _select_graph_with_highest_score(self, return_metrics: bool = False) -> SequentialWorkFlowGraph | tuple[SequentialWorkFlowGraph, Optional[dict]]:
        """Select the graph in `self._snapshot` with the highest score."""
        if len(self._snapshot) == 0:
            if return_metrics:
                return self.graph, None
            return self.graph
            
        snapshot_scores = [np.mean(list(snapshot["metrics"].values())) for snapshot in self._snapshot]
        best_index = np.argmax(snapshot_scores)

        if isinstance(self.graph, SequentialWorkFlowGraph):
            graph = SequentialWorkFlowGraph.from_dict(self._snapshot[best_index]["graph"])
        else:
            raise ValueError(f"Invalid graph type: {type(self.graph)}. The graph should be an instance of `SequentialWorkFlowGraph`.")
        
        if return_metrics:
            return graph, self._snapshot[best_index]["metrics"]
        return graph


    def _get_all_system_prompts(self) -> List[Variable]:
        """Gets all system prompts from the textgrad agents."""
        system_prompts = []
        for node in self.graph.nodes:
            system_prompts.append(node.textgrad_agent.system_prompt)
        return system_prompts

    
    def _get_all_prompt_templates(self) -> List[Variable]:
        """Gets all prompt templates from the textgrad agents."""
        prompt_templates = []
        for node in self.graph.nodes:
            prompt_templates.append(node.textgrad_agent.prompt_template)
        return prompt_templates

    
    def _add_missing_input_placeholder(self, prompt_template: str, node: str | WorkFlowNode) -> str:
        """If placeholders for the required inputs are missing in the node's prompt template, add them.
        Otherwise, returns the original prompt template.
        """
        if isinstance(node, str):
            node = self.graph.get_node(node)

        for input in node.inputs:
            if "<input>{" + input.name + "}</input>" not in prompt_template:
                if "{" + input.name + "}" in prompt_template:
                    # Only missing input tags
                    prompt_template = prompt_template.replace("{" + input.name + "}", "<input>{" + input.name + "}</input>")
                else:
                    # Missing both input tags and input placeholders
                    prompt_template += "\n\n" + input.description + ":\n<input>{" + input.name + "}</input>"
        
        return prompt_template


    def _create_output_lookup(self) -> dict[str, str]:
        """Creates a lookup table for output names to node names."""
        output_name_to_node_name = {}
        for node in self.graph.nodes:
            for output in node.outputs:
                output_name_to_node_name[output.name] = node.name
        return output_name_to_node_name
            

