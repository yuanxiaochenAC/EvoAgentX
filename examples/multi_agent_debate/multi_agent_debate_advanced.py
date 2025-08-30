import os
import random
from dotenv import load_dotenv

from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.models.model_configs import OpenAILLMConfig, OpenRouterConfig
from evoagentx.agents.customize_agent import CustomizeAgent


def create_dynamic_agent(role_name, role_description, model_config, temperature_adjustment=0.0):
    """动态创建智能体，根据角色调整模型参数"""
    
    # 基础提示词模板
    base_prompt = f"""
You are debater #{{agent_id}} (role: {role_name}). This is round {{round_index}} of {{total_rounds}}.

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
    adjusted_config = model_config.copy()
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
        prompt=base_prompt,
        llm_config=adjusted_config,
        inputs=inputs,
        outputs=outputs,
        parse_mode="xml",
    )


def create_role_model_combinations():
    """创建角色和模型的组合配置"""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # 定义角色
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
    
    # 角色-模型映射策略
    role_model_mapping = {
        # 乐观角色使用较温暖的模型
        "Optimist": ("gpt4o_mini", 0.1),  # 增加温度
        "Advocate": ("gpt4o", 0.2),
        "Innovator": ("gpt4o", 0.3),
        
        # 分析角色使用较冷静的模型
        "Analyst": ("llama", -0.1),  # 降低温度
        "Expert": ("gpt4o", -0.1),
        "Skeptic": ("llama", 0.0),
        
        # 其他角色使用默认配置
        "Pessimist": ("gpt4o_mini", 0.0),
        "Conservative": ("llama", -0.1),
        "Critic": ("gpt4o_mini", 0.0),
        "Mediator": ("gpt4o", 0.0),
    }
    
    return roles, models, role_model_mapping


def run_dynamic_combination():
    """运行动态组合的辩论"""
    print("=== 动态角色-模型组合辩论 ===")
    
    roles, models, mapping = create_role_model_combinations()
    
    # 随机选择5个角色
    selected_roles = random.sample(list(roles.keys()), 5)
    
    # 为每个角色创建对应的智能体
    agents = []
    for role in selected_roles:
        model_name, temp_adjust = mapping[role]
        model_config = models[model_name]
        agent = create_dynamic_agent(role, roles[role], model_config, temp_adjust)
        agents.append(agent)
    
    # 创建辩论图
    graph = MultiAgentDebateActionGraph(
        llm_config=OpenAILLMConfig(model="gpt-4o-mini", openai_key=os.getenv("OPENAI_API_KEY")),
        debater_agents=agents,
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
    
    # 显示配置信息
    print("\n智能体配置:")
    for i, agent in enumerate(agents):
        model_name = agent.llm_config.model if hasattr(agent.llm_config, 'model') else "Unknown"
        temp = agent.llm_config.temperature if hasattr(agent.llm_config, 'temperature') else "Unknown"
        print(f"  智能体 {i}: {agent.name} (模型: {model_name}, 温度: {temp})")


def run_balanced_debate():
    """运行平衡的辩论：确保正反两方都有代表"""
    print("\n=== 平衡辩论：正反两方代表 ===")
    
    roles, models, mapping = create_role_model_combinations()
    
    # 选择支持方和反对方的角色
    pro_roles = ["Optimist", "Advocate", "Innovator"]
    con_roles = ["Pessimist", "Skeptic", "Conservative"]
    neutral_roles = ["Analyst", "Mediator", "Expert"]
    
    # 创建平衡的智能体组合
    selected_roles = [
        random.choice(pro_roles),      # 支持方
        random.choice(con_roles),     # 反对方  
        random.choice(neutral_roles), # 中立方
        random.choice(pro_roles),     # 支持方
        random.choice(con_roles),     # 反对方
    ]
    
    agents = []
    for role in selected_roles:
        model_name, temp_adjust = mapping[role]
        model_config = models[model_name]
        agent = create_dynamic_agent(role, roles[role], model_config, temp_adjust)
        agents.append(agent)
    
    graph = MultiAgentDebateActionGraph(
        llm_config=OpenAILLMConfig(model="gpt-4o-mini", openai_key=os.getenv("OPENAI_API_KEY")),
        debater_agents=agents,
    )
    
    result = graph.execute(
        problem="Should we ban social media for teenagers? Give a final Yes/No with reasons.",
        num_agents=5,
        num_rounds=3,
        judge_mode="llm_judge",
        return_transcript=True,
    )
    
    print("最终答案:", result.get("final_answer"))
    print("获胜者:", result.get("winner"))
    
    print("\n辩论方阵:")
    for i, agent in enumerate(agents):
        stance = "支持方" if agent.name in pro_roles else "反对方" if agent.name in con_roles else "中立方"
        print(f"  智能体 {i}: {agent.name} ({stance})")


def run_model_comparison():
    """运行模型对比：相同角色使用不同模型"""
    print("\n=== 模型对比：相同角色不同模型 ===")
    
    roles, models, mapping = create_role_model_combinations()
    
    # 选择同一个角色，但使用不同模型
    role = "Analyst"
    role_description = roles[role]
    
    # 创建三个相同角色但不同模型的智能体
    agents = [
        create_dynamic_agent(f"{role}_GPT4o", role_description, models["gpt4o"], 0.0),
        create_dynamic_agent(f"{role}_GPT4oMini", role_description, models["gpt4o_mini"], 0.0),
        create_dynamic_agent(f"{role}_Llama", role_description, models["llama"], 0.0),
    ]
    
    graph = MultiAgentDebateActionGraph(
        llm_config=OpenAILLMConfig(model="gpt-4o-mini", openai_key=os.getenv("OPENAI_API_KEY")),
        debater_agents=agents,
    )
    
    result = graph.execute(
        problem="What's the best programming language for beginners? Give a final recommendation with reasons.",
        num_agents=3,
        num_rounds=2,
        judge_mode="llm_judge",
        return_transcript=True,
    )
    
    print("最终答案:", result.get("final_answer"))
    print("获胜者:", result.get("winner"))
    
    print("\n模型对比:")
    for i, agent in enumerate(agents):
        model_name = agent.llm_config.model if hasattr(agent.llm_config, 'model') else "Unknown"
        print(f"  智能体 {i}: {agent.name} (模型: {model_name})")


def main():
    """主函数"""
    print("多智能体辩论高级示例")
    print("=" * 50)
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置 OPENAI_API_KEY 环境变量")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("警告: 未设置 OPENROUTER_API_KEY 环境变量")
    
    # 运行动态组合
    run_dynamic_combination()
    
    # 运行平衡辩论
    run_balanced_debate()
    
    # 运行模型对比
    run_model_comparison()


if __name__ == "__main__":
    main()
