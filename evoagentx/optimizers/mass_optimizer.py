import numpy as np
import random
from tqdm import tqdm
from typing import List, Any, Optional, Callable
from pydantic import Field

from evoagentx.core.module import BaseModule
from evoagentx.models.base_model import BaseLLM
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.optimizers import MiproOptimizer
from evoagentx.utils.mipro_utils.register_utils import MiproRegistry

class MassOptimiser(BaseModule):
    workflow: Optional[list[Any]] = Field(default=None, description="The workflow to optimize.")
    trainset: Optional[List] = Field(default = None, description="The train set")
    valset: Optional[List] = Field(default=None, description="The validation set")
    executor_llm: Optional[BaseLLM] = Field(default=None, description="The LLM to use for execution.")
    optimizer_llm: Optional[BaseLLM] = Field(default=None, description="The LLM to use for optimization.")
    
    metric: Optional[Callable] = Field(default=None, description="Evaluation function used to assess the quality of generated results. Not required if `benchmark` is provided in `optimize` method. If provided, the `metric` should return a single float/bool value.")
    metric_threshold: Optional[float] = Field(default=None, description="Evaluate the quality of generated results. If the metric is greater than or equal to the threshold, the result is considered successful.")
    metric_instance: Optional[str] = Field(default=None, description="Metric instance to use. Only used when `benchmark` is provided in `optimize` method. The `evaluate` function within `benchmark` typically returns a dictionary of metrics. `metric_instance` specifies the key of the metric to use during optimization. If not provided, the first key will be used.")

    evaluator: Optional[Any] = Field(default=None, description="The evaluator to perform evaluation during optimization.")
    eval_rounds: int = Field(default=1, description="The number of rounds to evaluate the workflow.")
    max_steps: int = Field(default=30, description="The maximum number of steps to optimize the workflow.")

    save_interval: Optional[int] = Field(default=None, description="Save the workflow every save_interval steps.")
    save_path: Optional[str] = Field(default=None, description="Directory for saving logs, None means no saving")

    def init_module(self, **kwargs):
        self.total_calls = 0
        self.executor_llm_total_calls = 0
        self.rng = None

        if not self.workflow:
            raise ValueError("Workflow is required")

        if self.executor_llm is None and self.optimizer_llm is None:
            raise ValueError("Optimizer llm is required")

        if self.executor_llm is None:
            self.executor_llm = self.optimizer_llm

        if self.trainset is None:
            raise ValueError("Train set must be provided")

        self.valset = self.valset or self.trainset

    def optimize(self, 
                 *,
                 benchmark: Benchmark,
                 trainset: Optional[List] = None,
                 valset: Optional[List] = None,
                 softmax_temperature: float = 1.0,
                 configuration_rules: Callable = None,
                 agent_budget: int = 10):
        
        trainset = trainset or self.trainset
        valset = valset or self.valset
        self.benchmark = benchmark

        # Step  0: Optimize Predictor_0
        predictor = predictor(llm = self.executor_llm)

        predictor_registry = MiproRegistry()
        predictor_registry.track(predictor, "prompt", input_names=['problem'], output_names=['answer'])
        optimized_predictor = self._prompt_optimize()
        print(optimized_predictor.prompt)


        # 计算每个 block 的影响力
        incremental_influence = self._incremental_influence(trainset, valset)
        selection_probability = self._softmax_with_temperature(incremental_influence, softmax_temperature)
        # 随机采样，生成选择mask

        round = 0
        best_score = 0
        while round < self.max_steps:
            u = np.random.uniform(0, 1, size=selection_probability.shape)

            total = 0
            for ui, pi, blocki in zip(u, selection_probability, self.workflow.blocks):
                if ui <= pi:
                    blocki.activate = False
                
                space = blocki.search_space
                idx = random.randint(0, len(space) - 1)

                value = space[idx]
                total += value
                blocki.n = value

            if total >= agent_budget:
                continue

            workflow_score = self.evaluator(program = self.workflow,
                                            evalset = valset)
            
            if workflow_score > best_score:
                best_score = workflow_score
                best_workflow = self.workflow.deepcopy()
            
            self.workflow.reset()

        
        # STEP3: Optimize the entier workflow


    def _incremental_influence(self,trainset, valset):
        block_influence = []
        for block in self.workflow.blocks:
            # Step 0: evaluate each block before optimization
            score_before = self.evaluator(program = block,
                                          evalset = valset)

            # Step 1: prompt optimization for each block
            optimized_block = block

            # Step 2: obtain incremental influence
            score_after = self.evaluator(prgoram = optimized_block,
                                         evalset = valset)

            incremental_influence = score_after / score_before
            block_influence.append(incremental_influence)

        return block_influence

    def _softmax_with_temperature(self, logits, temperature):
        logits = np.array(logits, dtype=np.float32)
        logits = logits / temperature
        exps = np.exp(logits- np.max(logits))

        return exps / np.sum(exps)

    def _random_sample_searchSpace(self, workflow, agent_budget):
        values = []
        total = 0
        for block in workflow.blocks:
            space = block.search_space
            idx = random.randint(0, len(space) - 1)
            val = space[idx]
            total += val
            values.append(val)

            if total >= agent_budget:
                return [-1]
        return values
    
    def _prompt_optimize(self, registry, program, llm, max_bootstrapped_demos, max_labeled_demos, num_threads, eval_rounds, save_path):
        optimizer = MiproOptimizer(
            registry = registry,
            program = program,
            optimizer_llm = llm,
            max_bootstrapped_demos= max_bootstrapped_demos,
            max_labeled_demos = max_labeled_demos,
            num_threads = num_threads,
            eval_rounds= eval_rounds,
            auto = 'medium',
            save_path = save_path
        )

        optimizer.optimize(dataset = self.benchamrk)

        return optimizer.restore_best_program