#!/usr/bin/env python3
"""
MultiAgentDebate 高级示例 - 动态角色-模型匹配

这个示例专注于：
1. 根据角色特点动态选择最适合的模型
2. 通过角色设计优化辩论策略
3. 展示智能匹配对辩论质量的提升

与基础示例的区别：
- 基础示例：使用默认配置，展示基本功能
- 高级示例：自定义角色-模型匹配，优化辩论效果

与分组示例的区别：
- 分组示例：展示复杂架构设计
- 高级示例：展示角色和模型的智能匹配
"""

import os
import random
from dotenv import load_dotenv

from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.agents.customize_agent import CustomizeAgent


def create_optimized_agent(role_name, role_description, model_config, temperature_adjustment=0.0):
    """创建优化的智能体，根据角色特点调整模型参数"""
    
    # 根据角色特点调整prompt
    role_prompt = f"""
You are debater #{{agent_id}} (role: {{role}}). This is round {{round_index}} of {{total_rounds}}.

Problem:
{{problem}}

Conversation so far:
{{transcript_text}}

Instructions:
- You are a {role_name.upper()} who {role_description}
- Think briefly (<= 120 words), then present your {role_name.lower()} argument or rebuttal
- Focus on your unique perspective and expertise
- If confident, provide your current answer for this round
- Your output MUST follow this XML template:

<response>
  <thought>Your brief {role_name.lower()} reasoning</thought>
  <argument>Your {role_name.lower()} argument or rebuttal</argument>
  <answer>Optional current answer; leave empty if uncertain</answer>
</response>
"""
    
    # 调整模型配置
    adjusted_config = model_config.model_copy()
    if hasattr(adjusted_config, 'temperature'):
        adjusted_config.temperature = max(0.0, min(1.0, adjusted_config.temperature + temperature_adjustment))
    
    inputs = [
        {"name": "problem", "type": "str", "description": "Problem statement"},
        {"name": "transcript_text", "type": "str", "description": "Formatted debate transcript so far"},
        {"name": "role", "type": "str", "description": "Debater role/persona"},
        {"name": "agent_id", "type": "str", "description": "Debater id (string)"},
        {"name": "round_index", "type": "str", "description": "1-based round index"},
        {"name": "total_rounds", "type": "str", "description": "Total rounds"},
    ]
    
    outputs = [
        {"name": "thought", "type": "str", "description": "Brief reasoning", "required": True},
        {"name": "argument", "type": "str", "description": "Argument or rebuttal", "required": True},
        {"name": "answer", "type": "str", "description": "Optional current answer", "required": False},
    ]
    
    return CustomizeAgent(
        name=role_name,
        description=f"{role_name} debater: {role_description}",
        prompt=role_prompt,
        llm_config=adjusted_config,
        inputs=inputs,
        outputs=outputs,
        parse_mode="xml",
    )


def create_role_model_mapping():
    """创建角色-模型映射策略"""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # 定义角色及其特点
    roles = {
        "Optimist": "always sees the bright side and positive opportunities",
        "Pessimist": "focuses on risks, problems, and potential downsides", 
        "Analyst": "provides data-driven, balanced analysis",
        "Innovator": "thinks outside the box and suggests creative solutions",
        "Conservative": "values tradition, stability, and proven approaches",
        "Skeptic": "questions assumptions and demands evidence",
        "Advocate": "passionately defends a particular viewpoint",
        "Mediator": "seeks common ground and compromise",
        "Expert": "provides specialized knowledge and technical insights",
        "Critic": "identifies flaws and suggests improvements"
    }
    
    # 定义模型配置
    models = {
        "gpt4o_mini": OpenAILLMConfig(model="gpt-4o-mini", openai_key=openai_key, temperature=0.3),
        "gpt4o": OpenAILLMConfig(model="gpt-4o", openai_key=openai_key, temperature=0.2),
        "llama": OpenRouterConfig(model="meta-llama/llama-3.1-70b-instruct", openrouter_key=openrouter_key, temperature=0.3),
    }
    
    # 角色-模型匹配策略（基于角色特点选择最适合的模型）
    role_model_mapping = {
        # 需要创造力的角色使用GPT-4o
        "Innovator": ("gpt4o", 0.3),      # 高温度增加创造性
        "Advocate": ("gpt4o", 0.2),       # 中等温度增加说服力
        
        # 需要精确分析的角色使用Llama
        "Analyst": ("llama", -0.1),       # 低温度增加精确性
        "Expert": ("llama", 0.0),         # 标准温度保持专业性
        "Skeptic": ("llama", 0.0),       # 标准温度保持批判性
        
        # 其他角色使用GPT-4o-mini（平衡性能和成本）
        "Optimist": ("gpt4o_mini", 0.1),  # 轻微增加温度
        "Pessimist": ("gpt4o_mini", 0.0), # 标准温度
        "Conservative": ("gpt4o_mini", -0.1), # 降低温度增加稳定性
        "Critic": ("gpt4o_mini", 0.0),    # 标准温度
        "Mediator": ("gpt4o_mini", 0.1),  # 轻微增加温度增加灵活性
    }
    
    return roles, models, role_model_mapping


def run_optimized_debate():
    """运行优化的辩论：根据角色特点选择最适合的模型"""
    print("=== 优化辩论：角色-模型智能匹配 ===")
    
    roles, models, mapping = create_role_model_mapping()
    
    # 选择具有代表性的角色组合
    selected_roles = ["Analyst", "Innovator", "Skeptic", "Advocate", "Mediator"]
    
    # 为每个角色创建优化的智能体
    agents = []
    for role in selected_roles:
        model_name, temp_adjust = mapping[role]
        model_config = models[model_name]
        agent = create_optimized_agent(role, roles[role], model_config, temp_adjust)
        agents.append(agent)
    
    # 创建辩论图
    graph = MultiAgentDebateActionGraph(
        debater_agents=agents,
        llm_config=agents[0].llm_config if agents else None,
    )
    
    # 执行辩论
    result = graph.execute(
        problem="Should we invest heavily in AI research? Give a final Yes/No with reasons.",
        num_agents=5,
        num_rounds=3,
        judge_mode="llm_judge",
        return_transcript=True,
    )
    
    print("最终答案:", result.get("final_answer"))
    print("获胜者:", result.get("winner"))
    if result.get("winner_answer"):
        print("获胜者答案:", result.get("winner_answer"))
    
    # 显示角色-模型匹配信息
    print("\n角色-模型匹配策略:")
    for i, agent in enumerate(agents):
        model_name = agent.llm_config.model if hasattr(agent.llm_config, 'model') else "Unknown"
        temp = agent.llm_config.temperature if hasattr(agent.llm_config, 'temperature') else "Unknown"
        print(f"  {agent.name}: {model_name} (温度: {temp}) - {roles[agent.name]}")





def main():
    """主函数"""
    print("MultiAgentDebate 高级示例 - 动态角色-模型匹配")
    print("=" * 60)
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置 OPENAI_API_KEY 环境变量")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("警告: 未设置 OPENROUTER_API_KEY 环境变量")
    
    # 运行优化辩论
    run_optimized_debate()


if __name__ == "__main__":
    main()
