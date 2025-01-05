from ..core.registry import register_model
from .model_configs import LiteLLMConfig
from .base_model import BaseLLM


@register_model(config_cls=LiteLLMConfig, alias=["litellm"])
class LiteLLM(BaseLLM):
    pass
