import asyncio  
import os
import nest_asyncio

from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import VectorStoreConfig, DBConfig, StoreConfig
from evoagentx.rag.rag_config import (
    ReaderConfig, ChunkerConfig, IndexConfig, RetrievalConfig, EmbeddingConfig, RAGConfig
)
from evoagentx.core.message import MessageType
from evoagentx.agents.long_term_memory_agent import MemoryAgent

CORPUS_FILE = "./corpus_id.txt"  # 保存 corpus_id 的文件路径

async def test_memory_agent_auto_write_and_chat():
    # 初始化 LLM
    config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=os.environ["OPENAI_API_KEY"],
        temperature=0.1,
    )
    llm = OpenAILLM(config=config)

    # 初始化存储
    store_config = StoreConfig(
        dbConfig=DBConfig(db_name="sqlite", path="./debug/data/memory_test.sql"),
        vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=768, index_type="flat_l2"),
        graphConfig=None,
        path="./debug/data/memory_test_index"
    )
    storage_handler = StorageHandler(storageConfig=store_config)

    embedding = EmbeddingConfig(
        provider="huggingface",
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu"
    )

    rag_config = RAGConfig(
        reader=ReaderConfig(recursive=False, exclude_hidden=True),
        chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=0),
        embedding=embedding,
        index=IndexConfig(index_type="vector"),
        retrieval=RetrievalConfig(retrivel_type="vector", postprocessor_type="simple", top_k=2, similarity_cutoff=0.3)
    )

    # 初始化 MemoryAgent
    agent = MemoryAgent(
        llm=llm,
        rag_config=rag_config,
        storage_handler=storage_handler,
        name="MemoryAgent",
        description="Test auto memory write",
    )

    # ✅ 尝试加载已保存的 corpus_id
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, "r") as f:
            saved_corpus_id = f.read().strip()
        print(f"加载已有 corpus_id: {saved_corpus_id}")
        agent.long_term_memory.default_corpus_id = saved_corpus_id
    else:
        # 第一次运行，保存当前的 corpus_id
        with open(CORPUS_FILE, "w") as f:
            f.write(agent.long_term_memory.default_corpus_id)
        print(f"首次运行，保存 corpus_id: {agent.long_term_memory.default_corpus_id}")

    # 多轮对话
    questions = [
        "What is Python?",
        "Who created Python?",
        "What is AI?",
        "What can Python be used for?"
    ]

    for q in questions:
        response = await agent.async_chat(q)
        print(f"Q: {q}")
        print("Agent response:", response)

    # ✅ 使用 search 检查记忆是否写入
    all_memories = await agent.memory_manager.handle_memory(
        action="search",
        user_prompt="Python",
        top_k=10
    )

    queries = [m for m, _ in all_memories if m.msg_type in (MessageType.INPUT, MessageType.REQUEST)]
    responses = [m for m, _ in all_memories if m.msg_type == MessageType.RESPONSE]

    assert any("Python" in str(m.content) for m in queries), "用户输入未写入记忆"
    assert len(responses) > 0, "模型输出未写入记忆"

    print("✅ 多轮对话 + 自动写入记忆测试通过！")

if __name__ == "__main__":
    nest_asyncio.apply()  # 兼容已有事件循环
    asyncio.run(test_memory_agent_auto_write_and_chat())
