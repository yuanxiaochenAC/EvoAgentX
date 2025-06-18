import os 
from dotenv import load_dotenv
from typing import Any, Callable 
import asyncio
from evoagentx.workflow.operators import Predictor
from evoagentx.models import LiteLLMConfig, LiteLLM, OpenAILLMConfig, OpenAILLM 
from evoagentx.workflow.blocks.aggregate import aggregate
from evoagentx.workflow.blocks.reflect import reflect
from evoagentx.workflow.blocks.debate import debate
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def main():
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
    executor_llm = OpenAILLM(config=openai_config)

    # aggre = aggregate(executor_llm, 3)
    # print(aggre.execute(question = " what's the capital of france"))

    # refl = reflect(executor_llm, 2)
    # print(refl.execute(question = " what's the capital of france"))

    debater = debate(executor_llm, 3)

    # 创建一个debate例子
    question = "How to optimize the performance of a Python program?"

    # 初始解决方案列表
    solutions = [
        "Restarting the computer can solve all performance problems",
        "Change all code to print('hello world')",
        "Delete all comments, because comments slow down the program",
        "Use magic words: 'abracadabra' to optimize the code",
        "Change Python to JavaScript, because JavaScript is faster",
        "Change all variable names to single letters, like a, b, c",
        "Add import time; time.sleep(10) at the beginning of the code to let the program rest",
        "Change all functions to lambda expressions, no matter how complex",
        "Use a while True loop instead of all other loops",
        "Store all data in global variables for faster access"
    ]

    # 执行debate
    result = debater.execute(question=question, solutions=solutions)

    print(result)
    print("answer", result['answer'])
    print("\n----------------------------\n")
    print("index", result['index'])
    print("\n----------------------------\n")
    print("reasoning", result['reasoning'])

    # 测试Test operator
    from evoagentx.workflow.operators import Test
    from evoagentx.benchmark.humaneval import AFlowHumanEval

    # 创建Test operator
    test_operator = Test(executor_llm)

    # 示例代码和问题
    test_problem = "Write a function that adds two numbers"
    test_solution = '''python
def add_numbers(a, b):
    return a + b
'''
    test_entry_point = "add_numbers"

    # 创建benchmark (这里用HumanEval作为示例)
    benchmark = AFlowHumanEval()

    # 调用Test operator
    print("\n=== Testing Test Operator ===")
    test_result = await test_operator.async_execute(
        problem=test_problem,
        solution=test_solution,
        entry_point=test_entry_point,
        benchmark=benchmark
    )

    print("Test result:", test_result)
    print("Passed:", test_result['result'])
    print("Final solution:", test_result['solution'])

if __name__ == "__main__":
    asyncio.run(main())