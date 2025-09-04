
本教程系统讲解 EvoAgentX 的多智能体辩论 Multi-Agent Debate（简称 MAD）框架：核心概念、快速上手、关键参数、策略化角色与模型映射、分组工作流、以及配置的保存/加载。文中示例均可直接运行，并可参考 `examples/multi_agent_debate/` 下的示例脚本。

## 快速开始

最小可运行示例（自一致裁判或 LLM 裁判二选一）：

```python
from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.models import OpenAILLMConfig

debate = MultiAgentDebateActionGraph(
    name="MAD Minimal",
    description="Minimal runnable example for multi-agent debate",
    llm_config=OpenAILLMConfig(model="gpt-4o", temperature=0.5, max_completion_tokens=800),
)

result = debate.execute(
    problem="Should we invest heavily in AI research? Give a final Yes/No with reasons.",
    num_agents=5,
    num_rounds=5,
    judge_mode="llm_judge",          # 可选："self_consistency"
    return_transcript=True,
)

print("Final:", result.get("final_answer"))
print("Winner:", result.get("winner"))
```

更多可运行示例参见：

```12:27:examples/multi_agent_debate/multi_agent_debate.py
def run_self_consistency_example():
    llm_config = get_llm_config()
    debate = MultiAgentDebateActionGraph(
        name="MAD Minimal",
        description="Minimal runnable example for multi-agent debate",
        llm_config=llm_config,
    )
    fixed_problem = "How many labeled trees on 10 vertices ..."
    result = debate.execute(
        problem=fixed_problem,
        num_agents=3,
        num_rounds=5,
        judge_mode="self_consistency",
        return_transcript=True,
    )
```

## 核心概念

- 辩手（Debater）：承担论证与反驳的智能体，可设置不同 persona 与提示词。
- 裁判（Judge）：可选的 LLM 裁判，基于论辩记录判定胜者与最终答案；或启用自一致（self-consistency）。
- 回合（Rounds）：往返辩论的轮数。更多回合通常更稳健，但成本更高。
- 抄本（Transcript）：历史论辩的可控可见性，影响论证的上下文与信息量。

## execute() 关键参数与默认值建议

- problem: 必填。题目或任务描述。
- num_agents: 默认为 3。建议范围 3-7；过多易噪声与成本上升。
- num_rounds: 默认为 3。建议 2-6；任务复杂度↑可适当提高。
- judge_mode: "llm_judge" | "self_consistency"。开放问题推荐 "llm_judge"；客观题或可聚焦单一最终值时可用 "self_consistency"。
- return_transcript: 是否返回回合记录，便于审计与可视化。
- personas: 可选。自定义角色列表（名称/风格/目标）；若不提供将自动生成通用辩手。
- transcript_mode: "prev" | "all"。仅上一轮或全部历史回合喂给辩手。"prev" 成本低，"all" 信息更全。
- enable_pruning: 是否启用剪枝，减少不佳候选的干扰。默认 False；当 num_agents 或 num_rounds 较大时建议开启。

参数在 `examples/multi_agent_debate/README.md` 中也有简要说明，可配合本文选择合适的策略。

```96:112:examples/multi_agent_debate/README.md
### execute() 方法参数

- `problem`: 辩论问题
- `num_agents`: 参与辩论的智能体数量 (默认3)
- `num_rounds`: 辩论轮次 (默认3)
- `judge_mode`: 裁判模式 ("llm_judge" 或 "self_consistency")
- `personas`: 自定义角色列表
- `transcript_mode`: 记录访问模式 ("prev" 或 "all")
```

## 角色设计与模型映射（高级）

通过为不同角色匹配更合适的模型与温度，可显著提升多样性与有效性。参考：

```86:131:examples/multi_agent_debate/multi_agent_debate_advanced.py
def create_role_model_mapping():
    roles = {
        "Optimist": "always sees the bright side ...",
        "Pessimist": "focuses on risks ...",
        "Analyst": "data-driven, balanced analysis",
        "Innovator": "thinks outside the box ...",
        ...
    }
    models = {
        "gpt4o_mini": OpenAILLMConfig(..., temperature=0.3),
        "gpt4o": OpenAILLMConfig(..., temperature=0.2),
        "llama": OpenRouterConfig(..., temperature=0.3),
    }
    role_model_mapping = {
        "Innovator": ("gpt4o", 0.3),
        "Analyst": ("llama", -0.1),
        "Optimist": ("gpt4o_mini", 0.1),
        ...
    }
```

要点：

- 为“创造力”角色提高温度；为“分析/怀疑”角色降低温度。
- 成本受模型选择影响，可用较弱模型承担“背景噪声”角色，以节流。
- 将 `CustomizeAgent` 的 `parse_mode` 设为 `xml`，严格模板输出，利于结构化判决与可视化。

示例（创建优化辩手）：

```28:83:examples/multi_agent_debate/multi_agent_debate_advanced.py
def create_optimized_agent(role_name, role_description, model_config, temperature_adjustment=0.0):
    role_prompt = """
You are debater #{agent_id} (role: {role}). This is round {round_index} of {total_rounds}.
...
<response>
  <thought>...</thought>
  <argument>...</argument>
  <answer>...</answer>
</response>
"""
    adjusted_config = model_config.model_copy()
    adjusted_config.temperature = ...
    return CustomizeAgent(..., prompt=role_prompt, llm_config=adjusted_config, parse_mode="xml")
```

## 分组模式（Group Graphs）

当单个辩手本身需要由一个子团队（子图）组成时，启用分组图工作流：

```1:16:examples/multi_agent_debate/multi_agent_debate_group.py
class GroupOfManyGraph(ActionGraph):
    name: str = "GroupOfManyGraph"
    description: str = "Group with variable number of inner debaters"
    llm_config: OpenAILLMConfig
    num_inner: int = 3
```

运行方式：

```116:136:examples/multi_agent_debate/multi_agent_debate_group.py
group1 = GroupOfManyGraph(llm_config=llm_cfg, num_inner=3)
group2 = GroupOfManyGraph(llm_config=llm_cfg, num_inner=4)
...
debate = MultiAgentDebateActionGraph(
    group_graphs_enabled=True,
    group_graphs=[group1, group2],
    llm_config=llm_cfg,
)
result = debate.execute(
    problem="设计一个可扩展的多模态RAG系统评测方案...",
    num_agents=2,
    num_rounds=3,
    judge_mode="llm_judge",
)
```

建议：

- 用子图聚合多视角“子辩手”，输出单一合成观点与可选答案。
- 将子图的输出严格模板化（XML/JSON），便于主图整合。
- 当 `num_inner` 较大时，结合 `enable_pruning` 与较低 `transcript_mode` 降本增效。

## 抄本策略与成本控制

- transcript_mode = "prev"：仅喂入上一轮摘要，适合多轮长辩；可配合回合摘要节点降低 token 压力。
- transcript_mode = "all"：信息最全，但成本最高；仅在轮次较少、问题复杂时使用。
- 可选启用“简述再喂”（在辩手前插一个摘要 Agent），进一步压缩上下文。

## 剪枝与搜索深度

- enable_pruning: True 时，在每轮或整场结束时对候选进行裁剪，减少低质量分支。
- 建议与自一致结合：多样生成 → 剪枝 → 复核。

## 配置管理：保存、加载与复用

```78:94:examples/multi_agent_debate/README.md
loaded_debate = MultiAgentDebateActionGraph.load_module("my_debate_config.json")
new_debate = MultiAgentDebateActionGraph.from_dict(config_dict)
config = debate.get_config()
```

完整演示：

```52:86:examples/multi_agent_debate/config_methods_example.py
graph = MultiAgentDebateActionGraph(
    name="Demo Debate",
    description="演示用的辩论图",
    debater_agents=agents,
)
config = graph.get_config()
save_path = graph.save_module("demo_debate_config.json")
new_graph_from_dict = MultiAgentDebateActionGraph.from_dict(config)
new_graph_from_file = MultiAgentDebateActionGraph.load_module("demo_debate_config.json")
```

## 实战调优清单（经验值）

- 明确 judge_mode：客观题→自一致；开放题→LLM 裁判。
- 角色多样但不过量：3-5 个“互补视角”足够；多则噪声↑。
- 控温：创造性↑（0.6-0.9），分析性↓（0.1-0.3）。
- 抄本裁剪：长辩优先“prev”，必要时加摘要节点。
- 结构化输出：统一 XML/JSON 模板，降解析难度，提高裁判稳定性。
- 成本优先级：将“噪声角色”绑定低价模型，关键角色用强模型。

## 参考与入口

- 示例目录：`examples/multi_agent_debate/`
- 基础示例：`examples/multi_agent_debate/multi_agent_debate.py`
- 高级映射：`examples/multi_agent_debate/multi_agent_debate_advanced.py`
- 分组模式：`examples/multi_agent_debate/multi_agent_debate_group.py`
- 配置方法：`examples/multi_agent_debate/config_methods_example.py`


