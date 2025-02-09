from pydantic import Field
from typing import Optional, List

from ..core.logging import logger
from ..core.module import BaseModule
from ..core.base_config import Parameter
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput
from ..prompts.agent_generator import AGENT_GENERATION_ACTION

class AgentGenerationInput(ActionInput):

    goal: str = Field(description="A detailed statement of the workflow's goal, explaining the objectives the entire workflow aims to achieve")
    workflow: str = Field(description="An overview of the entire workflow, detailing all sub-tasks with their respective names, descriptions, inputs, and outputs")
    task: str = Field(description="A detailed JSON representation of the sub-task requiring agent generation. It should include the task's name, description, inputs, and outputs.")

    history: Optional[str] = Field(default=None, description="Optional field containing previously selected or generated agents.")
    suggestion: Optional[str] = Field(default=None, description="Optional suggestions to refine the generated agents.")
    existing_agents: Optional[str] = Field(default=None, description="Optional field containing the description of predefined agents, including each agent's name, role, and available actions.")
    tools: Optional[str] = Field(default=None, description="Optional field containing the description of tools that agents can use, including each tool's name and functionality.")


class GeneratedAgent(BaseModule):

    name: str 
    description: str 
    inputs: List[Parameter]
    outputs: List[Parameter]
    prompt: str
    tools: Optional[List[str]] = None


class AgentGenerationOutput(ActionOutput):

    selected_agents: List[str] = Field(description="A list of selected agent's names")
    generated_agents: List[GeneratedAgent] = Field(description="A list of generated agetns to address a sub-task")
    

class AgentGeneration(Action):

    def __init__(self, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else AGENT_GENERATION_ACTION["name"]
        description = kwargs.pop("description") if "description" in kwargs else AGENT_GENERATION_ACTION["description"]
        prompt = kwargs.pop("prompt") if "prompt" in kwargs else AGENT_GENERATION_ACTION["prompt"]
        inputs_format = kwargs.pop("inputs_format") if "inputs_format" in kwargs else AgentGenerationInput
        outputs_format = kwargs.pop("outputs_format") if "outputs_format" in kwargs else AgentGenerationOutput
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> AgentGenerationOutput:
        
        if not inputs:
            logger.error("AgentGeneration action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to AgentGeneration action is None or empty.')
        
        inputs_format: AgentGenerationInput = self.inputs_format
        outputs_format: AgentGenerationOutput = self.outputs_format

        prompt_params_names = inputs_format.get_attrs()
        prompt_params_values = {param: inputs.get(param, "") for param in prompt_params_names}
        prompt = self.prompt.format(**prompt_params_values)
        agents = llm.generate(
            prompt = prompt, 
            system_message = sys_msg, 
            parser=outputs_format
        )
        
        if return_prompt:
            return agents, prompt
        
        return agents
    