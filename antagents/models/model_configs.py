from pydantic import Field
from typing import Optional

import torch 

from ..core.base_config import BaseConfig


class LLMGenerationConfig(BaseConfig):
    
    # generation parameters 
    max_new_tokens: int = Field(default=1024, description="maximum number of generated tokens")
    temperature: float = Field(default=1.0, description="the temperature used to scaling logits")
    do_sample: bool = Field(default=False, description="whether use sampling to generate output")
    top_p: float = Field(default=1.0, description="only consider tokens with probabilities larger than top_p when sampling")


class LLMConfig(BaseConfig):

    llm_type: str
    generation_config: Optional[LLMGenerationConfig] = Field(default_factory=LLMGenerationConfig)


class APILLMConfig(LLMConfig):

    api_key: str 
    model_name: str 


def get_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class LocalLLMConfig(LLMConfig):

    model_name: str 
    device: Optional[str] = Field(default_factory=get_default_device)

