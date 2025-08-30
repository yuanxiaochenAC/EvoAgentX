"""
默认的多智能体辩论 LLM 配置池。

你可以直接修改本文件，或在此处改为从外部 YAML/JSON 加载。
优先级：若用户在运行时给出 agent_llm_configs，则以用户为准；否则使用这里的默认池。
"""

import os
from typing import List

from ...models.model_configs import (
    LLMConfig,
    OpenAILLMConfig,
    OpenRouterConfig,
    LiteLLMConfig,
)


def load_default_llm_config_pool() -> List[LLMConfig]:
    """返回默认的 LLM 配置池。

    如需改为从文件加载：
    - 读取环境变量 MAD_MODELS_PATH 指向的 YAML/JSON
    - 解析后构造对应的 LLMConfig 列表并返回
    这里提供一个环境变量驱动的简单默认实现。
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    pool: List[LLMConfig] = []

    if openai_key:
        pool.append(OpenAILLMConfig(model="gpt-4o-mini", openai_key=openai_key, temperature=0.3))
        pool.append(OpenAILLMConfig(model="gpt-4o", openai_key=openai_key, temperature=0.2))

    if openrouter_key:
        pool.append(OpenRouterConfig(model="meta-llama/llama-3.1-70b-instruct", openrouter_key=openrouter_key, temperature=0.3))

    # 如需本地/代理模型，可取消注释：
    # pool.append(LiteLLMConfig(model="ollama/llama3:instruct", temperature=0.3))

    return pool


