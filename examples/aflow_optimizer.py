from evoagentx.optimizers import AFlowOptimizer
from evoagentx.models import LiteLLMConfig, LiteLLM, OpenAILLMConfig, OpenAILLM 
from evoagentx.benchmark import AFlowHumanEval

OPENAI_API_KEY = "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" # OpenAI's KEY 
ANTHROPIC_API_KEY = "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" 

def main():

    claude_config = LiteLLMConfig(model="anthropic/claude-3-5-sonnet-20240620", anthropic_key=ANTHROPIC_API_KEY, stream=True)
    optimizer_llm = LiteLLM(config=claude_config)
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True) 
    executor_llm = OpenAILLM(config=openai_config)

    # load benchmark
    humaneval = AFlowHumanEval()

    # create optimizer
    optimizer = AFlowOptimizer(
        question_type="code", 
        graph_path = "examples/aflow/humaneval",
        optimized_path = "debug/aflow_optimized_humaneval",
        optimizer_llm=optimizer_llm,
        executor_llm=executor_llm,
        operators=["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"],
        eval_rounds=3,
        max_rounds=20, 
    )

    # run optimization
    optimizer.optimize(humaneval)

    # run test 
    optimizer.test(humaneval)


if __name__ == "__main__":
    main() 