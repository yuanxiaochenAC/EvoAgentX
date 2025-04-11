from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import OpenAI, Stream 
from openai.types.chat import ChatCompletion
from typing import Optional, List, Dict, Any
import json
import asyncio
from litellm import token_counter, cost_per_token
from ..core.registry import register_model
from .model_configs import OpenAILLMConfig
from .base_model import BaseLLM
from .model_utils import Cost, cost_manager, get_openai_model_cost 


@register_model(config_cls=OpenAILLMConfig, alias=["openai_llm"])
class OpenAILLM(BaseLLM):

    def init_model(self):
        config: OpenAILLMConfig = self.config
        self._client = OpenAI(api_key=config.openai_key)
        self._default_ignore_fields = ["llm_type", "output_response", "openai_key", "deepseek_key", "anthropic_key"] # parameters in OpenAILLMConfig that are not OpenAI models' input parameters 
        if self.config.model not in get_openai_model_cost():
            raise KeyError(f"'{self.config.model}' is not a valid OpenAI model name!")
        
        if self.config.tools:
            self._tool_callers = dict()
            for tool in self.config.tools:
                self._tool_callers[tool[1]["function"]["name"]] = tool[0]
            self.config.tools = [i[1] for i in self.config.tools]

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

    def update_completion_params(self, params1: dict, params2: dict) -> dict:
        config_params: list = self.config.get_config_params()
        for key, value in params2.items():
            if key in self._default_ignore_fields:
                continue
            if key not in config_params:
                continue
            params1[key] = value
        return params1

    def get_completion_params(self, **kwargs):
        completion_params = self.config.get_set_params(ignore=self._default_ignore_fields)
        completion_params = self.update_completion_params(completion_params, kwargs)
        return completion_params
    
    def get_stream_output(self, response: Stream, output_response: bool=True) -> str:
        output = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                if output_response:
                    print(content, end="", flush=True)
                output += content
        if output_response:
            print("")
        return output

    def get_completion_output(self, response: ChatCompletion, output_response: bool=True) -> str:
        output = response.choices[0].message.content
        if output_response:
            print(output)
        return output

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def single_generate(self, messages: List[dict], **kwargs) -> str:
        stream = kwargs["stream"] if "stream" in kwargs else self.config.stream
        output_response = kwargs["output_response"] if "output_response" in kwargs else self.config.output_response

        # try:
        #     completion_params = self.get_completion_params(**kwargs)
        #     response = self._client.chat.completions.create(messages=messages, **completion_params)
        #     if stream:
        #         output = self.get_stream_output(response, output_response=output_response)
        #         cost = self._stream_cost(messages=messages, output=output)
        #     else:
        #         output: str = self.get_completion_output(response=response, output_response=output_response)
        #         cost = self._completion_cost(response) # calculate completion cost
        #     self._update_cost(cost=cost)
        # except Exception as e:
        #     raise RuntimeError(f"Error during single_generate of OpenAILLM: {str(e)}")
        # return output
        
        
        try:
            completion_params = self.get_completion_params(**kwargs)
            response = self._client.chat.completions.create(messages=messages, **completion_params)
            
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                if stream:
                    output = self.get_stream_output(response, output_response=output_response)
                    cost = self._stream_cost(messages=messages, output=output)
                else:
                    output = self._handle_tool_calls(response, messages, **kwargs)
                    return output
            
            if stream:
                output = self.get_stream_output(response, output_response=output_response)
                cost = self._stream_cost(messages=messages, output=output)
            else:
                output = self.get_completion_output(response=response, output_response=output_response)
                cost = self._completion_cost(response)
            
            self._update_cost(cost=cost)
            return output
        except Exception as e:
            raise RuntimeError(f"Error during single_generate of OpenAILLM: {str(e)}")
        
    def batch_generate(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        return [self.single_generate(messages=one_messages, **kwargs) for one_messages in batch_messages]
    
    def _handle_tool_calls(self, response: ChatCompletion, messages: List[dict], **kwargs) -> str:
        output_response = kwargs.get("output_response", self.config.output_response)
        message = response.choices[0].message
        tool_calls = message.tool_calls
        
        cost = self._completion_cost(response)
        self._update_cost(cost=cost)
        
        conversation = messages.copy()
        
        assistant_message = {
            "role": "assistant",
            "content": message.content if message.content else None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in tool_calls
            ]
        }
        
        conversation.append(assistant_message)
        print("______________________")
        print("tool_calls: ", tool_calls)
        print("______________________")
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
                result = self._execute_tool_call(function_name, arguments)
                
                print("______________________COMPLETE______________________")
                print("result: ", result)
                print("______________________")
                
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result) if isinstance(result, dict) else str(result)
                })
            except Exception as e:
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error executing tool {function_name}: {str(e)}"
                })
        
        # Create a new kwargs dictionary without output_response
        kwargs_copy = kwargs.copy()
        if 'output_response' in kwargs_copy:
            del kwargs_copy['output_response']
            
        print("______________________")
        print("conversation: ", conversation)
        print("______________________")
        return self.single_generate(messages=conversation, output_response=output_response, **kwargs_copy)
    
    def _execute_tool_call(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        print("______________________")
        print("function_name: ", function_name)
        print("arguments: ", arguments)

        # First check if tools are provided in the config
        if not hasattr(self.config, 'tools') or not self.config.tools:
            return {
                "error": "No tools configured",
                "function": function_name,
                "arguments": arguments
            }
            
        # Check tools in tuple format (function, schema) - common for MCPToolkit tools
        for tool in self.config.tools:
            tool_name = tool['function'].get('name')
            if tool_name == function_name:
                try:
                    callback = self._tool_callers[function_name]
                    result = callback(**arguments)
                    return result
                except Exception as e:
                    print(f"Error executing tool {function_name}: {e}")
                    return {
                        "error": str(e),
                        "function": function_name,
                        "arguments": arguments
                    }

        # If we get here, the tool wasn't found
        return {
            "error": f"Tool '{function_name}' not found in configured tools",
            "function": function_name,
            "arguments": arguments
        }
    
    def _completion_cost(self, response: ChatCompletion) -> Cost:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return self._compute_cost(input_tokens=input_tokens, output_tokens=output_tokens)

    def _stream_cost(self, messages: List[dict], output: str) -> Cost:
        model: str = self.config.model
        input_tokens = token_counter(model=model, messages=messages)
        output_tokens = token_counter(model=model, text=output)
        return self._compute_cost(input_tokens=input_tokens, output_tokens=output_tokens)
    
    def _compute_cost(self, input_tokens: int, output_tokens: int) -> Cost:
        # use LiteLLM to compute cost, require the model name to be a valid model name in LiteLLM.
        input_cost, output_cost = cost_per_token(
            model=self.config.model, 
            prompt_tokens=input_tokens, 
            completion_tokens=output_tokens, 
        )
        cost = Cost(input_tokens=input_tokens, output_tokens=output_tokens, input_cost=input_cost, output_cost=output_cost)
        return cost
    
    def _update_cost(self, cost: Cost):
        cost_manager.update_cost(cost=cost, model=self.config.model)
    