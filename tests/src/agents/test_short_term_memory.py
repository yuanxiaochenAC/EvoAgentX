import asyncio  
import os
import nest_asyncio
from evoagentx.core.message import Message, MessageType
from evoagentx.agents.agent import Agent
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.actions.action import Action

# 定义一个 DummyAction，用于测试
class DummyAction(Action):
    def __init__(self, **kwargs):
        super().__init__(
            name="DummyAction",
            description="A test action that echoes input",
            **kwargs
        )

    def execute(self, llm, inputs, sys_msg=None, return_prompt=False, **kwargs):
        text = inputs.get("text", "")
        output = f"[Echo] {text}"
        prompt = f"Prompt used: {text}"
        if return_prompt:
            return output, prompt
        return output

class ShortTermMemoryTestAgent(Agent):
    """继承 Agent，用于短期记忆测试"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.short_term_memory.max_size = 5
        # ✅ 注意这里用 DummyAction()，不再传 action_name
        self.add_action(DummyAction())

async def test_short_term_memory_via_agent():
    # 初始化 LLM（虽然 DummyAction 不用 LLM，但 Agent 要求有）
    config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1,
    )
    llm = OpenAILLM(config=config)

    # 初始化 agent
    agent = ShortTermMemoryTestAgent(
        llm=llm,
        rag_config=None,
        storage_handler=None,
        name="STMTestAgent",
        description="Test short-term memory",
    )

    print("=== 添加单条消息 ===")
    await agent.async_execute("DummyAction", action_input_data={"text": "Hello 1"})
    for m in agent.short_term_memory.get():
        print(f"- {m.content}")

    print("\n=== 添加多条消息，超过 max_size 测试循环队列 ===")
    for i in range(2, 8):
        await agent.async_execute("DummyAction", action_input_data={"text": f"Msg {i}"})

    buffer_msgs = agent.short_term_memory.get()
    print("短期记忆内容（应最多 5 条）:")
    for i, m in enumerate(buffer_msgs, 1):
        print(f"{i}: {m.content}")

    # 检查 FIFO 行为（只看响应消息）
    responses = [m for m in buffer_msgs if isinstance(m.content, str) and m.content.startswith("[Echo]")]
    assert len(responses) <= agent.short_term_memory.max_size, "短期记忆超过 max_size"
    assert responses[0].content == "[Echo] Msg 5", "最旧响应消息应被覆盖"
    assert responses[-1].content == "[Echo] Msg 7", "最新响应消息应在队尾"

    print("\n✅ 短期记忆测试通过！")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(test_short_term_memory_via_agent())
