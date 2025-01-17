from typing import Optional, Type, List
from ..core.registry import register_model
from .model_configs import LiteLLMConfig
from .base_model import BaseLLM, LLMOutputParser


@register_model(config_cls=LiteLLMConfig, alias=["litellm"])
class LiteLLM(BaseLLM):

    def init_model(self):
        pass

    def single_generate(self, messages: List[dict], **kwargs) -> str:
        return "" 
    
    def batch_generate(self, messages: List[List[dict]], **kwargs) -> List[str]:
        return [""]
    
    def parse_generated_text(self, text: str, parser: Optional[Type[LLMOutputParser]]=None, **kwargs) -> LLMOutputParser:
        return LLMOutputParser.parse("")
    
    def parse_generated_texts(self, texts: List[str], parser: Optional[Type[LLMOutputParser]]=None, **kwargs) -> List[LLMOutputParser]:
        return [LLMOutputParser.parse("")]
