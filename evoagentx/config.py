from typing import Optional, Union, List
from .core.base_config import BaseConfig
from .models.model_configs import LLMConfig
from .core.registry import MODEL_REGISTRY


class Config(BaseConfig):

    llm_config: dict
    agents: Optional[List[Union[str, dict]]] = []

    def init_module(self):
        if isinstance(self.llm_config, dict):
            llm_config_data = self.llm_config
            llm_type = llm_config_data.get("llm_type", None)
            if llm_type is None:
                raise ValueError("must specify `llm_type` in in `llm_config`!")
            config_cls: LLMConfig = MODEL_REGISTRY.get_model_config(llm_type)
            self.llm_config = config_cls(**llm_config_data)
        if self.agents:
            for agent in self.agents:
                if isinstance(agent, dict) and "llm_config" not in agent:
                    agent["llm_config"] = self.llm_config
        
