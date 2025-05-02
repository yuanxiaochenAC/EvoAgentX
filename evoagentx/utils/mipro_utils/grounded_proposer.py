import random
import logging
import threading
from pydantic import Field
from typing import Any, Callable, Dict, Literal, Optional, Union, List, Tuple

from settings import settings
from evoagentx.workflow.workflow_graph import WorkflowGraph
from evoagentx.workflow.action_graph import ActionGraph
from evoagentx.core.module import BaseModule
from evoagentx.utils.mipro_utils.utils import get_source_code, strip_prefix
from evoagentx.utils.mipro_utils.dataset_summary_generator import create_dataset_summary



MAX_INSTRUCT_IN_HISTORY = 5  # 10

TIPS = {
        "none": "",
        "creative": "Don't be afraid to be creative when creating the new instruction!",
        "simple": "Keep the instruction clear and concise.",
        "description": "Make sure your instruction is very informative and descriptive.",
        "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
        "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
    }


logger = logging.getLogger("MIPRO")

class GroundedProposer(BaseModule):
    prompt_model: Any = Field(
        default=settings.lm,
        description="The prompt model to use for generating instructions"
    )
    program: Any = Field(
        description="The program to generate instructions for"
    )
    trainset: List = Field(
        description="Training dataset used for instruction generation"
    )
    view_data_batch_size: int = Field(
        default=10,
        description="Number of examples to show when creating dataset summary"
    )
    use_dataset_summary: bool = Field(
        default=True,
        description="Whether to use dataset summary for instruction generation"
    )
    program_aware: bool = Field(
        default=True,
        description="Whether to use program code for instruction generation"
    )
    use_task_demos: bool = Field(
        default=True,
        description="Whether to use task demonstrations for instruction generation"
    )
    num_demos_in_context: int = Field(
        default=3,
        description="Number of demonstrations to include in context"
    )
    use_instruct_history: bool = Field(
        default=True,
        description="Whether to use instruction history for generation"
    )
    use_tip: bool = Field(
        default=True,
        description="Whether to use prompting tips for instruction generation"
    )
    set_tip_randomly: bool = Field(
        default=True,
        description="Whether to randomly select prompting tips"
    )
    set_history_randomly: bool = Field(
        default=True,
        description="Whether to randomly select from instruction history"
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print verbose output"
    )
    rng: Any = Field(
        default_factory=lambda: random,
        description="Random number generator to use"
    )
    
    def init_module(self):
        if self.program_aware:
            try:
                self.program_code_string = get_source_code(self.program)    
                if self.verbose:
                    print("SOURCE CODE:",self.program_code_string)
            except Exception as e:
                print(f"Error getting source code: {e}.\n\nRunning without program aware proposer.")
                self.program_aware = False
                self.data_summary  = None
                
        self.data_summary = None
        if self.use_dataset_summary:
            try:
                self.data_summary = create_dataset_summary(
                    trainset=self.trainset, view_data_batch_size=self.view_data_batch_size, prompt_model=self.prompt_model,
                )
                if self.verbose:
                    print(f"DATA SUMMARY: {self.data_summary}")
            except Exception as e:
                print(f"Error getting data summary: {e}.\n\nRunning without data aware proposer.")
                self.use_dataset_summary = False
                print("")

    def propose_instructions_for_program(
        self,
        trainset,
        program,
        demo_candidates,
        trial_logs,
        N,
        T,
    ) -> list[str]:
        """This method is responsible for returning the full set of new instructions for our program, given the specified criteria."""
        proposed_instructions = {}      

        if self.set_history_randomly:
            # Randomly select whether or not we're using instruction history
            use_history = self.rng.random() < 0.5
            self.use_instruct_history = use_history
            if self.verbose:
                print(f"Use history T/F: {self.use_instruct_history}")

        if not demo_candidates:
            if self.verbose:
                print("No demo candidates provided. Running without task demos.")
            self.use_task_demos = False
            # When no demo candidates are provided, defailt to N
            num_demos = N
        else:
            num_demos = max(len(demo_candidates[0]), 1)

        # Create an instruction for each predictor 
        for pred_i, predictor in enumerate(program.agents()):
            for demo_set_i in range(num_demos):
                if pred_i not in proposed_instructions:
                    proposed_instructions[pred_i] = []
                selected_tip = None
                if self.set_tip_randomly:
                    if self.verbose:
                        print("Using a randomly generated configuration for our grounded proposer.")
                    # Randomly select the tip
                    selected_tip_key = self.rng.choice(list(TIPS.keys()))
                    selected_tip = TIPS[selected_tip_key]
                    self.use_tip = bool(
                        selected_tip,
                    )
                    if self.verbose:
                        print(f"Selected tip: {selected_tip_key}")

                proposed_instructions[pred_i].append(
                    self.propose_instruction_for_predictor(
                        program=program,
                        predictor=predictor,
                        pred_i=pred_i,
                        T=T,
                        demo_candidates=demo_candidates,
                        demo_set_i=demo_set_i,
                        trial_logs=trial_logs,
                        tip=selected_tip,
                    ),
                )
        
        return proposed_instructions
    
    def propose_instruction_for_predictor(
        self,
        program,
        predictor,
        pred_i,
        T,
        demo_candidates,
        demo_set_i,
        trial_logs,
        tip=None,
    ) -> str:
        """This method is responsible for returning a single instruction for a given predictor, using the specified criteria."""

        # Create an instruction history string for our predictor
        instruction_history = create_predictor_level_history_string(
            program, pred_i, trial_logs, MAX_INSTRUCT_IN_HISTORY,
        )

        # Create our instruction generator class (given specific criteria for this round of proposal)
        instruction_generator = GenerateModuleInstruction(
            program_code_string=self.program_code_string,
            use_dataset_summary=self.use_dataset_summary,
            program_aware=self.program_aware,
            use_task_demos=self.use_task_demos and demo_candidates,
            use_instruct_history=self.use_instruct_history and instruction_history,
            use_tip=self.use_tip,
            verbose=self.verbose
        )

        # Generate a new instruction for our predictor, using the temperature specified for this round
        original_temp = self.prompt_model.kwargs["temperature"]

        epsilon = self.rng.uniform(0.01, 0.05)
        modified_temp = T + epsilon

        with settings.context(lm=self.prompt_model):
            self.prompt_model.kwargs["temperature"] = modified_temp
            proposed_instruction = instruction_generator.forward(
                demo_candidates=demo_candidates,
                pred_i=pred_i,
                demo_set_i=demo_set_i,
                program=program,
                data_summary=self.data_summary,
                previous_instructions=instruction_history,
                num_demos_in_context = self.num_demos_in_context,
                tip=tip,
            ).proposed_instruction
        self.prompt_model.kwargs["temperature"] = original_temp

        # Log the trace used to generate the new instruction, along with the new instruction itself
        if self.verbose:
            self.prompt_model.inspect_history(n=1)
            print(f"PROPOSED INSTRUCTION: {proposed_instruction}")

        return strip_prefix(proposed_instruction)