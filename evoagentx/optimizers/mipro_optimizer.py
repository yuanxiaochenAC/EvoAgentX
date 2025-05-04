import logging
import random
import numpy as np
from pydantic import Field
from typing import Any, Callable, Dict, Literal, Optional, Union, List, Tuple

from ..evaluators.evaluator import Evaluator
from ..workflow.workflow import WorkFlowGraph
from ..workflow.action_graph import ActionGraph
from ..core.module import BaseModule
from ..models.base_model import BaseLLM
from ..utils.mipro_utils.settings import settings
from ..utils.mipro_utils.grounded_proposer import GroundedProposer
from ..utils.mipro_utils.utils import (create_minibatch,
                                      create_n_fewshot_demo_sets,
                                      )

logger = logging.getLogger("MIPRO")
logging.basicConfig(level=logging.INFO)

# Constants
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50

AUTO_RUN_SETTINGS = {
    "light": {"num_trials": 7, "val_size": 100},
    "medium": {"num_trials": 25, "val_size": 300},
    "heavy": {"num_trials": 50, "val_size": 1000},
}

# ANSI escape codes for colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default


class MiproOptimizer(BaseModule):

    graph: Optional[Union[WorkFlowGraph, ActionGraph]] = Field(default=None, description="The workflow to optimize.")
    grpah_path: Optional[str] = Field(default=None, description="Path to the workflow to optimize.")
    metric: Optional[Callable] = Field(default=None, description="Evaluation function used to assess the quality of generated results")
    prompt_model: BaseLLM = Field(default=settings.lm, description="Language model used for prompt generation")
    task_model: BaseLLM = Field(default=settings.lm, description="Language model used for task execution")
    teacher_settings: Dict = Field(default_factory=dict, description="Configuration parameters for the teacher model")
    max_bootstrapped_demos: int = Field(default=4, description="Maximum number of bootstrapped examples")
    max_labeled_demos: int = Field(default=4, description="Maximum number of labeled examples")
    auto: Optional[Literal["light", "medium", "heavy"]] = Field(default="medium", description="Automation level, can be light/medium/heavy")
    num_candidates: int = Field(default=10, description="Number of candidates generated per round")
    num_threads: Optional[int] = Field(default=1, description="Number of parallel threads, None means using all available threads")
    max_errors: int = Field(default=10, description="Maximum number of allowed errors")
    seed: int = Field(default=9, description="Random seed for reproducibility")
    init_temperature: float = Field(default=0.5, description="Initial sampling temperature")
    verbose: bool = Field(default=False, description="Whether to output detailed logs")
    track_stats: bool = Field(default=True, description="Whether to track statistics")
    log_dir: Optional[str] = Field(default=None, description="Directory for saving logs, None means no saving")
    metric_threshold: Optional[float] = Field(default=None, description="Metric threshold to stop optimization when reached")


    def init_module(self):
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.rng = None
        if not self.graph and not self.graph_path:
            raise ValueError("Either graph or graph_path must be provided")
        self.graph = self.graph or WorkFlowGraph.from_file(self.graph_path)
        
    def optimize(
        self,
        student: Any,
        *,
        trainset: List,
        trainset_inputs: List[str],
        teacher: Any = None,
        valset: Optional[List] = None,
        valset_inputs: Optional[List[str]] = None,
        num_trials: int = 30,
        max_bootstrapped_demos: Optional[int] = None,
        max_labeled_demos: Optional[int] = None,
        seed: Optional[int] = None,
        minibatch: bool = True,
        minibatch_size: int = 35,
        minibatch_full_eval_steps: int = 5,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        requires_permission_to_run: bool = True,
        provide_traceback: Optional[bool] = None,
    ) -> Any:
        
        seed = seed or self.seed
        self._set_random_seeds(seed)
        
        if max_bootstrapped_demos is not None:
            self.max_bootstrapped_demos = max_bootstrapped_demos
        
        if max_labeled_demos is not None:
            self.max_labeled_demos = max_labeled_demos
        
        
        if valset is not None and valset_inputs is None:
            raise ValueError("valset inputs must be provided")
        
        trainset, valset = self._set_and_validate_datasets(trainset, valset)
        
        
        zeroshot_opt = (self.max_bootstrapped_demos == 0) and (self.max_labeled_demos == 0)
        num_trials, valset, minibatch = self._set_hyperparams_from_run_mode(
            student, num_trials, minibatch, zeroshot_opt, valset
        )
        
        if self.auto:
            self._print_auto_run_settings(num_trials, minibatch, valset)

        if minibatch and minibatch_size > len(valset):
            raise ValueError(f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}.")

        program = student.deepcopy()
        # TODO: 敲定evaluator的细节
        evaluate = Evaluator(self.task_model)
        
        # Step 1: Bootstrap few-shot examples
        demo_candidates = self._bootstrap_fewshot_examples(program, trainset, trainset_inputs, seed, teacher)

        # Step 2: Propose instruction candidates
        instruction_candidates = self._propose_instructions(
            program,
            trainset,
            demo_candidates,
            view_data_batch_size,
            program_aware_proposer,
            data_aware_proposer,
            tip_aware_proposer,
            fewshot_aware_proposer,
        )

        # If zero-shot, discard demos
        if zeroshot_opt:
            demo_candidates = None

        # Step 3: Find optimal prompt parameters
        best_program = self._optimize_prompt_parameters(
            program,
            instruction_candidates,
            demo_candidates,
            evaluate,
            valset,
            num_trials,
            minibatch,
            minibatch_size,
            minibatch_full_eval_steps,
            seed,
        )

        return best_program

            
    def _set_and_validate_datasets(self, trainset: List, valset: Optional[List]):
        if not trainset:
            raise ValueError("Trainset cannot be empty.")

        if valset is None:
            if len(trainset) < 2:
                raise ValueError("Trainset must have at least 2 examples if no valset specified.")
            valset_size = min(1000, max(1, int(len(trainset) * 0.80)))
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff]
        else:
            if len(valset) < 1:
                raise ValueError("Validation set must have at least 1 example.")

        return trainset, valset

    def _set_random_seeds(self, seed: int):
        self.rng = random.Random(seed)
        np.random.seed(seed)
    
    def _set_hyperparams_from_run_mode(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        zeroshot_opt: bool,
        valset: List,
    ) -> Tuple[int, List, bool]:
        if self.auto is None:
            return num_trials, valset, minibatch

        num_vars = len(program.agents())
        if not zeroshot_opt:
            num_vars *= 2  # Account for few-shot examples + instruction variables

        auto_settings = AUTO_RUN_SETTINGS[self.auto]
        num_trials = auto_settings["num_trials"]
        valset = create_minibatch(valset, batch_size=auto_settings["val_size"], rng=self.rng)
        minibatch = len(valset) > MIN_MINIBATCH_SIZE
        self.num_candidates = int(np.round(np.min([num_trials * num_vars, (1.5 * num_trials) / num_vars])))

        return num_trials, valset, minibatch

    def _print_auto_run_settings(self, num_trials: int, minibatch: bool, valset: List):
        logger.info(
            f"\nRUNNING WITH THE FOLLOWING {self.auto.upper()} AUTO RUN SETTINGS:"
            f"\nnum_trials: {num_trials}"
            f"\nminibatch: {minibatch}"
            f"\nnum_candidates: {self.num_candidates}"
            f"\nvalset size: {len(valset)}\n"
        )

    def _bootstrap_fewshot_examples(self, program: Any, trainset: List, trainset_inputs: List[str], seed: int, teacher: Any) -> Optional[List]:
        logger.info("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
        if self.max_bootstrapped_demos > 0:
            logger.info(
                "These will be used as few-shot example candidates for our program and for creating instructions.\n"
            )
        else:
            logger.info("These will be used for informing instruction proposal.\n")

        logger.info(f"Bootstrapping N={self.num_candidates} sets of demonstrations...")

        zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0
        
        try:
            demo_candidates = create_n_fewshot_demo_sets(
                student=program,
                num_candidate_sets=self.num_candidates,
                trainset=trainset,
                trainset_inputs=trainset_inputs,
                max_labeled_demos=(LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_labeled_demos),
                max_bootstrapped_demos=(
                    BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_bootstrapped_demos
                ),
                metric=self.metric,
                max_errors=self.max_errors,
                teacher=teacher,
                teacher_settings=self.teacher_settings,
                seed=seed,
                metric_threshold=self.metric_threshold,
                rng=self.rng,
            )
        except Exception as e:
            logger.info(f"Error generating few-shot examples: {e}")
            logger.info("Running without few-shot examples.")
            demo_candidates = None

        return demo_candidates
    
    def _propose_instructions(
        self,
        program: Any,
        trainset: List,
        demo_candidates: Optional[List],
        view_data_batch_size: int,
        program_aware_proposer: bool,
        data_aware_proposer: bool,
        tip_aware_proposer: bool,
        fewshot_aware_proposer: bool,
    ) -> Dict[int, List[str]]:
        logger.info("\n==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==")
        logger.info(
            "We will use the few-shot examples from the previous step, a generated dataset summary, a summary of the program code, and a randomly selected prompting tip to propose instructions."
        )

        proposer = GroundedProposer(
            program=program,
            trainset=trainset,
            prompt_model=self.prompt_model,
            view_data_batch_size=view_data_batch_size,
            program_aware=program_aware_proposer,
            use_dataset_summary=data_aware_proposer,
            use_task_demos=fewshot_aware_proposer,
            num_demos_in_context=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
            use_tip=tip_aware_proposer,
            set_tip_randomly=tip_aware_proposer,
            use_instruct_history=False,
            set_history_randomly=False,
            verbose=self.verbose,
            rng=self.rng,
        )

        logger.info("\nProposing instructions...\n")
        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            N=self.num_candidates,
            T=self.init_temperature,
            trial_logs={},
        )

        for i, agent in enumerate(program.agents()):
            logger.info(f"Proposed Instructions for Predictor {i}:\n")
            instruction_candidates[i][0] = agent['prompt']
            for j, instruction in enumerate(instruction_candidates[i]):
                logger.info(f"{j}: {instruction}\n")
            logger.info("\n")

        return instruction_candidates
