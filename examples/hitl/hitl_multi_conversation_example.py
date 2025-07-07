import asyncio
import os
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.hitl import HITLOutsideConversationAgent, HITLManager
# from evoagentx.workflow import WorkFlow, WorkFlowGraph
# from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.agents import AgentManager
# from evoagentx.core.base_config import Parameter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def main():
    """
    演示如何使用HITLOutsideConversationAgent来修改WorkFlow的JSON结构
    """
    print("🚀 EvoAgentX HITL Outside Conversation Agent 示例")
    print("=" * 80)
    
    # 配置LLM
    llm_config = OpenAILLMConfig(
        model="gpt-4o", 
        openai_key=OPENAI_API_KEY, 
        stream=True, 
        output_response=True
    )
    llm = OpenAILLM(llm_config)
    
    # 创建HITLOutsideConversationAgent
    conversation_agent = HITLOutsideConversationAgent(
        name="WorkFlowModificationAgent",
        description="支持多轮对话来修改WorkFlow的JSON结构",
        llm_config=llm_config
    )
    
    # 激活HITL功能
    hitl_manager = HITLManager()
    hitl_manager.activate()
    
    # 创建Agent管理器
    agent_manager = AgentManager(agents=[conversation_agent])
    
    # 示例1：从JSON文件加载并修改WorkFlow
    print("\n📋 示例1: 从JSON文件加载并修改WorkFlow")
    print("-" * 50)
    
    # 指定JSON文件路径
    json_file_path = "examples/output/tetris_game/workflow_demo_4o_mini.json"
    
    try:
        # 执行多轮对话修改
        
        result = await conversation_agent.conversation_action.async_execute(
            llm=llm,
            inputs={"workflow_json_path": json_file_path},
            hitl_manager=hitl_manager
        )
        
        print("\n✅ 从JSON文件修改WorkFlow成功!")
        print(f"修改后的WorkFlow: {result[0]['final_workflow']}")
        
    except Exception as e:
        print(f"❌ 从JSON文件修改WorkFlow失败: {e}")
    
    # # 示例2：从现有WorkFlow实例修改
    # print("\n📋 示例2: 从现有WorkFlow实例修改")
    # print("-" * 50)
    
    # try:
    #     # 创建一个简单的WorkFlow实例
    #     nodes = [
    #         WorkFlowNode(
    #             name="task1",
    #             description="第一个任务",
    #             inputs=[Parameter(name="input1", type="string", description="输入参数")],
    #             outputs=[Parameter(name="output1", type="string", description="输出参数")]
    #         ),
    #         WorkFlowNode(
    #             name="task2", 
    #             description="第二个任务",
    #             inputs=[Parameter(name="input2", type="string", description="输入参数")],
    #             outputs=[Parameter(name="output2", type="string", description="输出参数")]
    #         )
    #     ]
        
    #     edges = [
    #         WorkFlowEdge(source="task1", target="task2")
    #     ]
        
    #     graph = WorkFlowGraph(
    #         goal="示例工作流",
    #         nodes=nodes,
    #         edges=edges
    #     )
        
    #     existing_workflow = WorkFlow(graph=graph, llm=llm)
        
    #     # 执行多轮对话修改
    #     result = await conversation_agent.actions[0].async_execute(
    #         llm=llm,
    #         inputs={"existing_workflow": existing_workflow},
    #         hitl_manager=hitl_manager
    #     )
        
    #     print("\n✅ 从现有WorkFlow实例修改成功!")
    #     print(f"修改后的WorkFlow: {result[0]['final_workflow']}")
        
    # except Exception as e:
    #     print(f"❌ 从现有WorkFlow实例修改失败: {e}")
    
    # 关闭HITL功能
    hitl_manager.deactivate()
    
    print("\n🎉 示例运行完成!")


if __name__ == "__main__":
    asyncio.run(main())
