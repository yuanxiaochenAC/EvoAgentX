import numpy as np
from pydantic import Field 
from copy import deepcopy
from typing import Literal, Union, Optional, List

from .optimizer import Optimizer
from ..core.logging import logger
from ..benchmark.benchmark import Benchmark
from ..workflow.action_graph import ActionGraph
from ..workflow.workflow_graph import SequentialWorkFlowGraph
from ..prompts.workflow.sem_optimizer import mutation_prompt, thinking_style

VALID_SCHEMES = ["python", "yaml", "code", "core", "bpmn"]

class SEMWorkFlowScheme:

    """
    The scheme of the workflow for SEM optimizer.
    """
    def __init__(self, graph: SequentialWorkFlowGraph, **kwargs):
        self.graph = graph # the workflow graph to be transformed
        self.kwargs = kwargs

    def convert_to_scheme(self, scheme: str) -> str:
        """
        Transform the WorkflowGraph to the desired scheme.
        """
        if scheme not in VALID_SCHEMES:
            raise ValueError(f"Invalid scheme: {scheme}. The scheme should be one of {VALID_SCHEMES}.") 
        if scheme == "python":
            repr = self.get_workflow_python_repr()
        elif scheme == "yaml":
            repr = self.get_workflow_yaml_repr()
        elif scheme == "code":
            repr = self.get_workflow_code_repr()
        elif scheme == "core":
            repr = self.get_workflow_core_repr()
        elif scheme == "bpmn":
            repr = self.get_workflow_bpmn_repr()
        return repr

    def parse_from_scheme(self, scheme: str, repr: str) -> SequentialWorkFlowGraph:
        """
        Parse the SequentialWorkFlowGraph from the given scheme and representation.
        """
        pass # TODO: implement the workflow parsing

    def _get_workflow_repr_info(self) -> List[dict]:
        """
        Get the information for the workflow representation.
        """
        info = []
        for node in self.graph.nodes:
            task_name = node.name
            input_names = [param.name for param in node.inputs] 
            output_names = [param.name for param in node.outputs]
            task_info = {
                "task_name": task_name,
                "input_names": input_names,
                "output_names": output_names
            }
            info.append(task_info)
        return info
    
    def _convert_to_func_name(self, name: str) -> str:
        """
        Convert the task name to the function name.
        """
        name = name.lower().strip()
        name = name.replace(' ', '_').replace('-', '_')
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        # Replace multiple consecutive underscores with a single underscore
        while '__' in name:
            name = name.replace('__', '_')
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name
    
    def _convert_to_title(self, name: str) -> str:
        func_name = self._convert_to_func_name(name)
        words = func_name.split('_')
        return ' '.join(word.capitalize() for word in words)
    
    def get_workflow_python_repr(self) -> str: 
        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""
        
        python_workflow_info = [] 
        for task_info in repr_info:
            name = self._convert_to_func_name(task_info['task_name'])
            input_names = [f'{input_name}' for input_name in task_info['input_names']]
            output_names = [f'{output_name}' for output_name in task_info['output_names']]
            python_workflow_info.append(
                "{{'name': '{name}', 'args': {args}, 'outputs': {outputs}}}".format(
                    name=name,
                    args=input_names,
                    outputs=output_names
                )
            )
        python_workflow_repr = "steps = [\n" + ",\n".join(python_workflow_info) + "\n]"
        return python_workflow_repr
    
    def get_workflow_yaml_repr(self) -> str:
        repr_info = self._get_workflow_repr_info() 
        if not repr_info:
            return ""
        
        yaml_workflow_info = []
        for task_info in repr_info:
            name = self._convert_to_func_name(task_info['task_name'])
            input_names = "\n".join([f'    - {input_name}' for input_name in task_info['input_names']])
            output_names = "\n".join([f'    - {output_name}' for output_name in task_info['output_names']])
            yaml_workflow_info.append(
                "- name: {name}\n  args:\n{input_names}\n  outputs:\n{output_names}".format(
                    name=name,
                    input_names=input_names,
                    output_names=output_names
                )
            )
        yaml_workflow_repr = "\n\n".join(yaml_workflow_info)
        return yaml_workflow_repr

    def get_workflow_code_repr(self) -> str:
        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""
            
        workflow_lines = []
        for task_info in repr_info:
            # Convert task name to snake_case
            name = self._convert_to_func_name(task_info['task_name'])
            
            # Format inputs and outputs
            inputs = ", ".join(task_info['input_names'])
            outputs = ", ".join(task_info['output_names'])
            
            # Create the line in format: task_name(inputs) -> outputs
            line = f"{name}({inputs}) -> {outputs}"
            workflow_lines.append(line)
            
        # Join all lines with newlines
        workflow_repr = "\n".join(workflow_lines)
        
        return workflow_repr

    def get_workflow_bpmn_repr(self) -> str:

        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""
            
        # Start the BPMN XML
        bpmn_lines = [
            '<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">',
            '<process id="software_dev_workflow" isExecutable="true">',
            '    <startEvent id="start" />'
        ]
        
        # Add tasks
        for i, task_info in enumerate(repr_info):
            task_name = self._convert_to_func_name(task_info['task_name'])
            task_title = self._convert_to_title(task_info['task_name'])
            bpmn_lines.append(f'    <task id="{task_name}" name="{task_title}" />')
            
        bpmn_lines.append('    <endEvent id="end" />')
        bpmn_lines.append('')
        bpmn_lines.append('    <!-- Workflow connections -->')
        
        # Add sequence flows
        # First flow from start to first task
        if repr_info:
            first_task_id = self._convert_to_func_name(repr_info[0]['task_name'])
            bpmn_lines.append(f'    <sequenceFlow id="flow1" sourceRef="start" targetRef="{first_task_id}" />')
            
        # Flows between tasks
        for i in range(len(repr_info) - 1):
            source_id = self._convert_to_func_name(repr_info[i]['task_name'])
            target_id = self._convert_to_func_name(repr_info[i + 1]['task_name'])
            flow_num = i + 2
            bpmn_lines.append(f'    <sequenceFlow id="flow{flow_num}" sourceRef="{source_id}" targetRef="{target_id}" />')
            
        # Last flow from last task to end
        if repr_info:
            last_task_id = self._convert_to_func_name(repr_info[-1]['task_name'])
            flow_num = len(repr_info) + 1
            bpmn_lines.append(f'    <sequenceFlow id="flow{flow_num}" sourceRef="{last_task_id}" targetRef="end" />')
            
        # Close tags
        bpmn_lines.append('</process>')
        bpmn_lines.append('</definitions>')
        
        return '\n'.join(bpmn_lines)
    
    def get_workflow_core_repr(self) -> str:

        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""
            
        workflow_lines = []
        for i, task_info in enumerate(repr_info, 1):
            # Convert task name to title case
            task_name = self._convert_to_title(task_info['task_name'])
            # Create the line with the specified format
            next_step = i + 1
            line = f"Step {i}::: Process ::: {task_name}:::next::Step {next_step}"
            workflow_lines.append(line)
            
        # Add the terminal step
        last_step = len(repr_info) + 1
        workflow_lines.append(f"Step {last_step}::: Terminal ::: End of Workflow:::")
        
        return "\n".join(workflow_lines)


class SimplePromptBreeder:
    """
    The simple prompt breeder for SEM optimizer.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate_mutation_prompt(self, task_description: str, **kwargs) -> str:
        """
        Generate the mutation prompt for optimization.
        """
        pass # TODO: implement the mutation prompt generation

    def generate_prompt(self, order: Literal["zero-order", "first-order"], **kwargs) -> str:
        """
        Generate the prompt for optimization.
        """
        pass # TODO: implement the prompt generation


class SEMOptimizer(Optimizer):

    graph: Union[SequentialWorkFlowGraph, ActionGraph] = Field(description="The workflow to optimize.")
    repr_scheme: str = Field(default="python", description="The scheme to represent the workflow.")
    optimize_mode: Literal["all", "structure", "prompt"] = Field(default="all", description="The mode to optimize the workflow.")
    order: Literal["zero-order", "first-order"] = Field(default="zero-order", description="Whether to use zero-order (using hyper-mutation prompt) or first-order (using mutation prompt) optimization.")

    def init_module(self, **kwargs):
        self._snapshot: List[dict] = []
        self._prompt_breeder = SimplePromptBreeder() # generate prompt for optimization
        if isinstance(self.graph, ActionGraph):
            if self.optimize_mode != "prompt":
                raise ValueError(
                    f"{type(self).__name__} only support prompt optimization when `graph` is an `ActionGraph`. "
                    "The `optimize_mode` should be set to `prompt`, but got {self.optimize_mode}."
                )

    def optimize(self, dataset: Benchmark, **kwargs):
        """
        Optimize the workflow.
        """
        logger.info(f"Optimizing the workflow with {self.repr_scheme} representation.")
        graph: Union[SequentialWorkFlowGraph, ActionGraph] = deepcopy(self.graph)
        logger.info(f"Run initial evaluation on the original workflow ...")
        metrics = self.evaluate(dataset, eval_mode="dev", graph=graph)
        logger.info(f"Initial metrics: {metrics}")
        self.log_snapshot(graph=graph, metrics=metrics)

        for i in range(self.max_steps):
            try:
                # perform a step of optimization
                graph = self.step()
                # evaluate the workflow
                if (i + 1) % self.eval_every_n_steps == 0:
                    logger.info(f"Evaluate the workflow at step {i+1} ...")
                    metrics = self.evaluate(dataset, eval_mode="dev", graph=graph)
                    logger.info(f"Step {i+1} metrics: {metrics}")
                    self.log_snapshot(graph=graph, metrics=metrics)
            except Exception as e:
                logger.warning(f"Error in step {i}: {e}. Skip this step.")
                continue
            if self.convergence_check():
                logger.info(f"Convergence check passed at step {i+1}. Stop the optimization.")
                break
        
        if i == self.max_steps - 1:
            logger.info(f"Reach the maximum number of steps {self.max_steps}. Stop the optimization.")
        
        # set self.graph to the best graph
        logger.info(f"Restore the best graph from the snapshot ...")
        self.restore_best_graph()
    
    def step(self, **kwargs) -> Union[SequentialWorkFlowGraph, ActionGraph]:
        """
        Take a step of optimization and return the optimized graph.
        """
        graph = self._select_graph_with_highest_score(return_metrics=False)
        if isinstance(graph, SequentialWorkFlowGraph):
            new_graph = self._workflow_graph_step(graph)
        elif isinstance(graph, ActionGraph):
            new_graph = self._action_graph_step(graph)
        else:
            raise ValueError(f"Invalid graph type: {type(graph)}. The graph should be an instance of `WorkFlowGraph` or `ActionGraph`.")
        return new_graph
    
    def evaluate(
        self, 
        dataset: Benchmark, 
        eval_mode: str = "test", 
        graph: Optional[Union[SequentialWorkFlowGraph, ActionGraph]] = None,
        indices: Optional[List[int]] = None,
        sample_k: Optional[int] = None,
        **kwargs
    ) -> dict:
        """
        Evaluate the workflow. If `graph` is provided, use the provided graph for evaluation. Otherwise, use the graph in the optimizer. 
        
        Args:
            dataset (Benchmark): The dataset to evaluate the workflow on.
            eval_mode (str): The evaluation mode. Choices: ["test", "dev", "train"].
            graph (Union[WorkFlowGraph, ActionGraph], optional): The graph to evaluate. If not provided, use the graph in the optimizer.
            indices (List[int], optional): The indices of the data to evaluate the workflow on.
            sample_k (int, optional): The number of data to evaluate the workflow on. If provided, a random sample of size `sample_k` will be used.
        
        Returns:
            dict: The metrics of the workflow evaluation.
        """
        graph = graph if graph is not None else self.graph
        metrics_list = []
        for _ in range(self.eval_rounds):
            metrics = self.evaluator.evaluate(
                graph=graph, 
                benchmark=dataset, 
                eval_mode=eval_mode, 
                indices=indices, 
                sample_k=sample_k,
                **kwargs
            )
            metrics_list.append(metrics)
        avg_metrics = self.evaluator._calculate_average_score(metrics_list)
        
        return avg_metrics
    
    def log_snapshot(self, graph: Union[SequentialWorkFlowGraph, ActionGraph], metrics: dict):
        """
        Log the snapshot of the workflow.
        """
        self._snapshot.append(
            {
                "index": len(self.snapshot),
                "graph": graph,
                "metrics": metrics,
            }
        )

    def _select_graph_with_highest_score(self, return_metrics: bool = False) -> Union[SequentialWorkFlowGraph, ActionGraph, dict]:
        """
        Select the graph in `self._snapshot` with the highest score.
        """
        snapshot_scores = [np.mean(snapshot["metrics"].values()) for snapshot in self._snapshot]
        best_index = np.argmax(snapshot_scores)
        graph = self._snapshot[best_index]["graph"]
        if return_metrics:
            return graph, self._snapshot[best_index]["metrics"]
        return graph
    
    def restore_best_graph(self):
        """
        Restore the best graph from the snapshot and set it to `self.graph`.
        """
        best_graph, best_metrics = self._select_graph_with_highest_score(return_metrics=True)
        logger.info(f"Restore the best graph from snapshot with metrics {best_metrics} ...")
        self.graph = best_graph

    def _wfg_structure_optimization_step(self, graph: SequentialWorkFlowGraph) -> SequentialWorkFlowGraph:
        """
        optinize the structure of the workflow graph and return the optimized graph.
        """
        pass # TODO: implement the workflow optimization step

    def _wfg_prompt_optimization_step(self, graph: SequentialWorkFlowGraph) -> SequentialWorkFlowGraph:
        """
        optinize the prompt of the workflow graph and return the optimized graph.
        """
        pass # TODO: implement the prompt optimization step
    
    def _workflow_graph_step(self, graph: SequentialWorkFlowGraph) -> SequentialWorkFlowGraph:
        """
        Take a step of optimization on the workflow graph and return the optimized graph.
        """
        if self.optimize_mode == "structure" or self.optimize_mode == "all":
            # optimize the structure of the graph    
            graph = self._wfg_structure_optimization_step(graph)
        if self.optimize_mode == "prompt" or self.optimize_mode == "all":
            # optimize the prompt of the graph
            graph = self._wfg_prompt_optimization_step(graph)
        
        return graph

    def _action_graph_step(self, graph: ActionGraph) -> ActionGraph:
        """
        Take a step of optimization on the action graph and return the optimized graph.
        """
        pass # TODO: implement the action graph step    
    
    