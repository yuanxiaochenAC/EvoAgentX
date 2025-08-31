# Multi-Agent Debate 使用指南

Multi-Agent Debate 是一个多智能体辩论框架，支持多个AI代理进行辩论并得出共识。

## 快速开始

### 1. 基础辩论示例

最简单的使用方式，使用默认配置：

```python
from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph

# 创建辩论系统
debate = MultiAgentDebateActionGraph()

# 执行辩论
result = debate.execute(
    problem="人工智能是否会对人类就业产生负面影响？",
    num_agents=3,
    num_rounds=2
)

print(f"最终答案: {result['final_answer']}")
print(f"获胜者: Agent #{result['winner']}")
```

### 2. 自定义Agent示例

参考 `multi_agent_debate.py` 学习如何创建自定义的辩手和裁判：

```python
from evoagentx.agents.customize_agent import CustomizeAgent

# 创建自定义辩手
debater_agent = CustomizeAgent(
    name="专业辩手",
    prompt="你是一位专业的辩手...",
    # ... 其他配置
)

# 创建自定义裁判
judge_agent = CustomizeAgent(
    name="公正裁判", 
    prompt="你是一位公正的裁判...",
    # ... 其他配置
)

# 使用自定义agent
debate = MultiAgentDebateActionGraph(
    debater_agents=[debater_agent],
    judge_agent=judge_agent
)
```

### 3. 高级功能示例

参考 `multi_agent_debate_advanced.py` 学习高级功能：

- **动态角色-模型匹配**: 根据角色自动选择最适合的LLM模型
- **辩论策略优化**: 通过不同的模型组合提升辩论质量

### 4. 分组模式示例

参考 `multi_agent_debate_group.py` 学习如何使用分组模式：

- **工作流图模式**: 使用 `group_graphs_enabled=True` 启用分组辩论
- **多层级辩论**: 每个辩手位置可以由一个包含多个子辩手的工作流图替代

```python
# 启用分组模式
debate = MultiAgentDebateActionGraph(
    group_graphs_enabled=True,
    group_graphs=[group_graph1, group_graph2]
)
```

### 5. 保存和加载配置

参考 `config_methods_example.py` 学习如何保存和加载辩论配置：

```python
# 保存配置
debate.save_module("my_debate_config.json")

# 加载配置
loaded_debate = MultiAgentDebateActionGraph.load_module("my_debate_config.json")

# 从字典创建
new_debate = MultiAgentDebateActionGraph.from_dict(config_dict)

# 获取配置
config = debate.get_config()
```

## 主要参数说明

### execute() 方法参数

- `problem`: 辩论问题
- `num_agents`: 参与辩论的智能体数量 (默认3)
- `num_rounds`: 辩论轮次 (默认3)
- `judge_mode`: 裁判模式 ("llm_judge" 或 "self_consistency")
- `personas`: 自定义角色列表
- `transcript_mode`: 记录访问模式 ("prev" 或 "all")

### 高级功能

- **剪枝功能**: 使用 `enable_pruning=True` 启用候选答案筛选
- **异步执行**: 使用 `async_execute()` 进行异步辩论
- **分组模式**: 使用 `group_graphs_enabled=True` 启用工作流图模式

## 示例文件说明

- `multi_agent_debate.py`: 基础使用示例
- `multi_agent_debate_advanced.py`: 高级功能示例
- `multi_agent_debate_group.py`: 分组模式示例
- `config_methods_example.py`: 配置方法示例

## 环境要求

确保设置了必要的API密钥：
```bash
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"  # 可选
```

## 下一步

1. 运行基础示例了解基本用法
2. 尝试自定义agent创建个性化辩手
3. 探索高级功能如动态模型匹配
4. 学习分组模式进行多层级辩论
5. 学习如何保存和复用辩论配置
