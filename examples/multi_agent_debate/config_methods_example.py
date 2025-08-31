#!/usr/bin/env python3
"""
MultiAgentDebateActionGraph 配置方法示例
展示如何使用 save_module, load_module, from_dict, get_config 方法
"""

import os
from dotenv import load_dotenv
from evoagentx.frameworks.multi_agent_debate.debate import MultiAgentDebateActionGraph
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.agents.customize_agent import CustomizeAgent

# 加载环境变量
load_dotenv()

def create_sample_agents():
    """创建示例agents"""
    agents = []
    
    # 创建第一个agent
    agent1 = CustomizeAgent(
        name="OptimistAgent",
        description="乐观的辩手，总是看到问题的积极面",
        prompt="你是一个乐观的辩手。请从积极的角度分析问题：{problem}",
        llm_config=OpenAILLMConfig(
            model="gpt-4o-mini",
            openai_key=os.getenv("OPENAI_API_KEY")
        ),
        inputs=[{"name": "problem", "type": "str", "description": "问题"}],
        outputs=[{"name": "argument", "type": "str", "description": "论点"}],
        parse_mode="title"
    )
    agents.append(agent1)
    
    # 创建第二个agent
    agent2 = CustomizeAgent(
        name="PessimistAgent", 
        description="悲观的辩手，总是看到问题的消极面",
        prompt="你是一个悲观的辩手。请从消极的角度分析问题：{problem}",
        llm_config=OpenAILLMConfig(
            model="gpt-4o-mini",
            openai_key=os.getenv("OPENAI_API_KEY")
        ),
        inputs=[{"name": "problem", "type": "str", "description": "问题"}],
        outputs=[{"name": "argument", "type": "str", "description": "论点"}],
        parse_mode="title"
    )
    agents.append(agent2)
    
    return agents

def demo_save_and_load():
    """演示保存和加载功能"""
    print("=== 演示保存和加载功能 ===")
    
    # 1. 创建辩论图
    agents = create_sample_agents()
    graph = MultiAgentDebateActionGraph(
        name="Demo Debate",
        description="演示用的辩论图",
        debater_agents=agents,
        llm_config=agents[0].llm_config if agents else None,
    )
    
    # 2. 获取配置
    print("\n1. 获取当前配置...")
    config = graph.get_config()
    print(f"配置包含 {len(config)} 个字段")
    print(f"Agent数量: {len(config.get('debater_agents', []))}")
    
    # 3. 保存配置
    print("\n2. 保存配置到文件...")
    save_path = graph.save_module("demo_debate_config.json")
    print(f"配置已保存到: {save_path}")
    
    # 4. 从字典创建新实例
    print("\n3. 从配置字典创建新实例...")
    new_graph_from_dict = MultiAgentDebateActionGraph.from_dict(config)
    print(f"新实例名称: {new_graph_from_dict.name}")
    print(f"新实例Agent数量: {len(new_graph_from_dict.debater_agents or [])}")
    
    # 5. 从文件加载新实例
    print("\n4. 从文件加载新实例...")
    new_graph_from_file = MultiAgentDebateActionGraph.load_module("demo_debate_config.json")
    print(f"加载的实例名称: {new_graph_from_file.name}")
    print(f"加载的实例Agent数量: {len(new_graph_from_file.debater_agents or [])}")
    
    # 6. 加载到现有实例
    print("\n5. 加载配置到现有实例...")
    empty_graph = MultiAgentDebateActionGraph()
    empty_graph.load_module("demo_debate_config.json")
    print(f"加载后实例名称: {empty_graph.name}")
    print(f"加载后Agent数量: {len(empty_graph.debater_agents or [])}")

def demo_error_handling():
    """演示错误处理"""
    print("\n=== 演示错误处理 ===")
    
    # 1. 尝试加载不存在的文件
    print("\n1. 尝试加载不存在的文件...")
    try:
        MultiAgentDebateActionGraph.load_module("nonexistent_file.json")
    except FileNotFoundError as e:
        print(f"预期的错误: {e}")
    
    # 2. 尝试从无效字典创建实例
    print("\n2. 尝试从无效字典创建实例...")
    try:
        invalid_config = {"invalid_field": "invalid_value"}
        MultiAgentDebateActionGraph.from_dict(invalid_config)
        print("成功创建实例（使用默认值）")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    print("MultiAgentDebateActionGraph 配置方法演示")
    print("=" * 50)
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置OPENAI_API_KEY环境变量")
        print("请设置环境变量后运行此示例")
    else:
        demo_save_and_load()
        demo_error_handling()
        
        print("\n=== 演示完成 ===")
        print("生成的文件:")
        print("- demo_debate_config.json (主配置文件)")
        print("- demo_debate_config_agents.json (Agent池文件)")
