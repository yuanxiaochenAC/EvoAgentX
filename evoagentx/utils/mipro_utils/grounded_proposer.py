import random
import logging
from pydantic import Field
from typing import Any, List, Optional
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.utils.mipro_utils.settings import settings
from evoagentx.core.module import BaseModule
from evoagentx.utils.mipro_utils.utils import get_source_code, strip_prefix, create_predictor_level_history_string, create_example_string
from evoagentx.utils.mipro_utils.dataset_summary_generator import create_dataset_summary

import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


MAX_INSTRUCT_IN_HISTORY = 5  # 10

TIPS = {
        "none": "", 
        "creative": "Don't be afraid to be creative when creating the new instruction!",
        "simple": "Keep the instruction clear and concise.",
        "description": "Make sure your instruction is very informative and descriptive.",
        "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
        "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
    }

describe_program = CustomizeAgent(
    name="DescribeProgram", 
    description="An agent that describes a program's functionality",
    prompt="""
    Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work.
    
    Program Code:
    {program_code}
    
    Program Example:
    {program_example}
    """,
    inputs = [
        {"name": "program_code", "type": "str", "description": "Pseudocode for a language model program designed to solve a particular task"},
        {"name": "program_example", "type": "str", "description": "An example of the program in use"}
    ],
    outputs = [
        {"name": "program_description", "type": "str", "description": "Describe what task the program is designed to solve, and how it goes about solving this task"}
    ],
    llm_config=OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY),
    parse_mode='str',
)

describe_module = CustomizeAgent(
    name="DescribeModule",
    description="An agent that describes a module's functionality in a program",
    prompt="""
    Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of one of the specified module in this pipeline.
    
    Program Code:
    {program_code}
    
    Program Example:
    {program_example}
    
    Program Description:
    {program_description}
    
    Module:
    {module}
    
    """,
    inputs = [
        {"name": "program_code", "type": "str", "description": "Pseudocode for a language model program designed to solve a particular task"},
        {"name": "program_example", "type": "str", "description": "An example of the program in use"},
        {"name": "program_description", "type": "str", "description": "Summary of the task the program is designed to solve, and how it goes about solving it"},
        {"name": "module", "type": "str", "description": "The module in the program that we want to describe"}
    ],
    outputs = [
        {"name": "module_description", "type": "str", "description": "Description of the module's role in the broader program"}
    ],
    llm_config=OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY),
    parse_mode='str',
)


def generate_instruction_class(
    use_dataset_summary=True,
    program_aware=True,
    use_task_demos=True,
    use_instruct_history=True,
    use_tip=True,
):
    inputs = []
    if use_dataset_summary:
        inputs.append({
            "name": "dataset_description",
            "type": "str", 
            "description": "A description of the dataset that we are using."
        })
    if program_aware:
        inputs.extend([
            {
                "name": "program_code",
                "type": "str",
                "description": "Language model program designed to solve a particular task."
            },
            {
                "name": "program_description", 
                "type": "str",
                "description": "Summary of the task the program is designed to solve, and how it goes about solving it."
            },
            {
                "name": "module",
                "type": "str", 
                "description": "The module to create an instruction for."
            },
            {
                "name": "module_description",
                "type": "str",
                "description": "Description of the module to create an instruction for."
            }
        ])
    inputs.append({
        "name": "task_demos",
        "type": "str",
        "description": "Example inputs/outputs of our module."
    })
    if use_instruct_history:
        inputs.append({
            "name": "previous_instructions",
            "type": "str",
            "description": "Previous instructions we've attempted, along with their associated scores."
        })
    inputs.append({
        "name": "basic_instruction",
        "type": "str",
        "description": "Basic instruction."
    })
    if use_tip:
        inputs.append({
            "name": "tip",
            "type": "str",
            "description": "A suggestion for how to go about generating the new instruction."
        })

    return CustomizeAgent(
        name="GenerateSingleModuleInstruction",
        description="An agent that generates instructions for language model modules",
        prompt="""Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task.""",
        inputs=inputs,
        outputs=[
            {
                "name": "proposed_instruction",
                "type": "str",
                "description": "Propose an instruction that will be used to prompt a Language Model to perform this task."
            }
        ],
        llm_config=settings.lm,
        parse_mode='str'
    )

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


class GenerateModuleInstruction(BaseModule):
    program_code_string: Optional[str] = Field(default=None, description="The string version of the program code.")
    use_dataset_summary: bool = Field(default=True)
    program_aware: bool = Field(default=False)
    use_task_demos: bool = Field(default=True)
    use_instruct_history: bool = Field(default=True)
    use_tip: bool = Field(default=True)
    verbose: bool = Field(default=False)

    def init_module(self):
        self.describe_program = describe_program
        self.describe_module = describe_module
        self.generate_module_instruction = generate_instruction_class(
            use_dataset_summary=self.use_dataset_summary,
            program_aware=self.program_aware,
            use_task_demos=self.use_task_demos,
            use_instruct_history=self.use_instruct_history,
            use_tip=self.use_tip,
        )

    def optimize(
        self,
        demo_candidates,
        pred_i,
        demo_set_i,
        program,
        previous_instructions,
        data_summary,
        num_demos_in_context=3,
        tip=None,
    ):
        def gather_examples_from_sets(candidate_sets, max_examples):
            """Helper function to gather up to augmented examples from given sets."""
            count = 0
            for candidate_set in candidate_sets:
                for example in candidate_set:
                    if "augmented" in example.keys():
                        cur_module = program.agents()[pred_i]
                        fields_to_use = cur_module["inputs"] + cur_module["outputs"]
                        yield create_example_string(fields_to_use, example)
                        count += 1
                        if count >= max_examples:
                            return

            basic_instruction = program.agents()[pred_i]['prompt']
            task_demos = ""
            
            if self.use_task_demos:
                adjacent_sets = (
                    [demo_candidates[pred_i][demo_set_i]] +
                    demo_candidates[pred_i][demo_set_i + 1:] +
                    demo_candidates[pred_i][:demo_set_i]
                )

                example_strings = gather_examples_from_sets(adjacent_sets, num_demos_in_context)
                task_demos = "\n\n".join(example_strings) + "\n\n"

            # Default to no demos provided if no examples were gathered, or if we're using the first demo set
            if not task_demos.strip() or demo_set_i == 0:
                task_demos = "No task demos provided."

            # Summarize the program
            program_description = "Not available"
            module_code = "Not provided"
            module_description = "Not provided"
            if self.program_aware:
                try:
                    program_description = strip_prefix(
                        self.describe_program(
                            program_code=self.program_code_string, program_example=task_demos,
                        ).program_description,
                    )
                    if self.verbose:
                        print(f"PROGRAM DESCRIPTION: {program_description}")

                    inputs = []
                    outputs = []
                    input_fields = cur_module.inputs_format.get_attrs()
                    module_code = f"{program.predictors()[pred_i].__class__.__name__}({', '.join(inputs)}) -> {', '.join(outputs)}"

                    module_description = self.describe_module(
                        program_code=self.program_code_string,
                        program_description=program_description,
                        program_example=task_demos,
                        module=module_code,
                        max_depth=10,
                    ).module_description
                except:
                    if self.verbose:
                        print("Error getting program description. Running without program aware proposer.")
                    self.program_aware = False

            # Generate an instruction for our chosen module
            if self.verbose:
                print(f"task_demos {task_demos}")

            instruct = self.generate_module_instruction(
                dataset_description=data_summary,
                program_code=self.program_code_string,
                module=module_code,
                program_description=program_description,
                module_description=module_description,
                task_demos=task_demos,
                tip=tip,
                basic_instruction=basic_instruction,
                previous_instructions=previous_instructions,
            )

            proposed_instruction = strip_prefix(instruct.proposed_instruction)

            return dspy.Prediction(proposed_instruction=proposed_instruction)
                
                
