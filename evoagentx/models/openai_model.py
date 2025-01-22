from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import OpenAI
from typing import Optional, List
from ..core.registry import register_model
from .model_configs import OpenAILLMConfig
from .base_model import BaseLLM


@register_model(config_cls=OpenAILLMConfig, alias=["openai_llm"])
class OpenAILLM(BaseLLM):

    def init_model(self):
        config: OpenAILLMConfig = self.config
        self._client = OpenAI(api_key=config.openai_key)

    def formulate_messages(self, prompts: List[str], system_messages: Optional[List[str]] = None) -> List[List[dict]]:
        
        if system_messages:
            assert len(prompts) == len(system_messages), f"the number of prompts ({len(prompts)}) is different from the number of system_messages ({len(system_messages)})"
        else:
            system_messages = [None] * len(prompts)
        
        messages_list = [] 
        for prompt, system_message in zip(prompts, system_messages):
            messages = [] 
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)

        return messages_list

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:

        config: OpenAILLMConfig = self.config
        set_params = config.get_set_params(ignore=["openai_key"])
        response = self._client.chat.completions.create(messages=messages, **set_params, **kwargs)
        if config.stream:
            output = ""
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    output += content
            print("")
        else:
            output: str = response.choices[0].message.content
            print(output)
        
        return output.strip()
        
    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        return [self.single_generate(messages=one_messages, **kwargs) for one_messages in batch_messages]
    
