import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLM, OpenAILLMConfig
from evoagentx.benchmark import HotPotQA 
from evoagentx.workflow import SequentialWorkFlowGraph
from evoagentx.core.callbacks import suppress_logger_info 
from evoagentx.optimizers.mipro_optimizer import MiproOptimizer
from evoagentx.evaluators.mipro_evaluator import Evaluate as mipro_evaluator
from evoagentx.utils.mipro_utils.settings import settings
from evoagentx.workflow.workflow import WorkFlowGraph
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class HotPotQASplits(HotPotQA):
    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._dev_data))
        full_test_data = self._dev_data 
        # randomly select 50 samples for dev and 100 samples for test
        self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._test_data = [full_test_data[idx] for idx in permutation[50:150]]

def collate_func(example: dict) -> dict:
    context_list = []
    for item in example["context"]:
        context = "Title: {}\nText: {}".format(item[0], " ".join([t.strip() for t in item[1]]))
        context_list.append(context)
    context = "\n\n".join(context_list)
    problem = "Context: {}\n\nQuestion: {}\n\nAnswer:".format(context, example["question"])
    return {"problem": problem}


    
    
    
hotpotqa_graph_data = {
    "goal": "Answer the question based on the context. The answer should be a direct response to the question, without including explanations or reasoning.",
    "tasks": [
        {
            "name": "answer_generate",
            "description": "Answer the question based on the context.",
            "inputs": [
                {"name": "problem", "type": "str", "required": True, "description": "The problem to solve."}
            ],
            "outputs": [
                {"name": "answer", "type": "str", "required": True, "description": "The answer to the problem."}
            ],
            "prompt": "Think step by step to answer the question. You should explain your thinking process in the 'thought' field, and provide the final answer in the 'answer' field.\nFormat your output in xml format, such as <thought>xxx</thought> and <answer>xxx</answer>.\n\nProblem: {problem}",
            "parse_mode": "xml"
        }
    ] 
}

def main():
    # Initialize LLM
    llm_config = OpenAILLMConfig(model="gpt-4", openai_key=OPENAI_API_KEY)
    llm = OpenAILLM(config=llm_config)
    settings.lm = llm_config

    # Load benchmark
    benchmark = HotPotQASplits()
    benchmark._load_data()
    trainset = [{"problem": collate_func(example)["problem"], "answer": example["answer"]} for example in benchmark._train_data]
    testset = [{"problem": collate_func(example)["problem"], "answer": example["answer"]} for example in benchmark._test_data]
    
    # Load workflow
    workflow_graph = SequentialWorkFlowGraph.from_dict(hotpotqa_graph_data)
    
    
    def evaluate_metric(example, prediction, trace=None):
        label = example["answer"]
        
        result = benchmark.evaluate(prediction, label)
        
        return result["f1"] >= 0.66
    
    # Create MIPRO evaluator
    evaluate = mipro_evaluator(
        devset=testset,
        metric=evaluate_metric,
        num_threads=32,
        display_progress=True,
        display_table=False,
    )

    # Create MIPRO optimizer
    optimizer = MiproOptimizer(
        graph=workflow_graph,
        metric=evaluate_metric,
        prompt_model=llm,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_candidates=15,
        auto="medium",
        num_threads=32,
        log_dir="examples/mipro/output/logs",
    )

    # Run optimization
    with suppress_logger_info():
        best_program = optimizer.optimize(
            trainset=trainset,
            with_inputs={"problem":"problem"},
        )
        
    output_path = r"C:\Users\31646\Desktop\EvoAgentX\examples\mipro\output\best_program_hotpotqa.json"
    best_program.save_module(output_path)
    # 保存评估结果
    with suppress_logger_info():
        post_results = evaluate(program = WorkFlowGraph.from_file(output_path), 
                                with_inputs={"problem":"problem"})
    

if __name__ == "__main__":
    main() 