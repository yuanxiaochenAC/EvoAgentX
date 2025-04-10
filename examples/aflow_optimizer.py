from evoagentx.config import Config 
from evoagentx.optimizers.aflow_optimizer import AFlowOptimizer
from evoagentx.models.openai_model import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark.humaneval import AFlowHumanEval

def main():

    config = Config.from_file("debug/config_template.json")
    llm_config = OpenAILLMConfig.from_dict(config.llm_config)
    llm = OpenAILLM(config=llm_config)

    # load benchmark
    humaneval = AFlowHumanEval()

    # create optimizer
    optimizer = AFlowOptimizer(
        question_type="code", 
        graph_path = "examples/aflow/humaneval",
        optimized_path = "debug/aflow_optimized_humaneval",
        optimizer_llm=llm,
        operators=["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"],
        eval_rounds=1,
        max_rounds=3, 
    )

    # run optimization
    optimizer.optimize(humaneval)

    # run test 
    optimizer.test(humaneval)


if __name__ == "__main__":
    main() 