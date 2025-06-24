import os 
import json
from dotenv import load_dotenv
from typing import Any

from evoagentx.benchmark import MATH
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.optimizers import MiproOptimizer
from evoagentx.utils.mipro_utils.register_utils import MiproRegistry
from evoagentx.workflow.operators import Predictor, Summarizer
from evoagentx.workflow.blocks.summarize import summarize
from evoagentx.workflow.blocks.aggregate import aggregate
from evoagentx.workflow.blocks.reflect import reflect
from evoagentx.workflow.blocks.debate import debate
from evoagentx.workflow.blocks.execute import execute

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_BOOTSTRAPPED_DEMOS = 1
MAX_LABELED_DEMOS = 0
AUTO = "light"
NUM_THREADS = 16
EVALUATION_ROUNDS = 1
class MathSplits(MATH):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # radnomly select 50 samples for training and 100 samples for test
        # self._train_data = [full_test_data[idx] for idx in permutation[:50]]
        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._test_data = [full_test_data[idx] for idx in permutation[100:200]]

    # define the input keys. 
    # If defined, the corresponding input key and value will be passed to the __call__ method of the program, 
    # i.e., program.__call__(**{k: v for k, v in example.items() if k in self.get_input_keys()})
    # If not defined, the program will be executed with the entire input example, i.e., program.__call__(**example)
    def get_input_keys(self):
        return ["problem"]
    
    # the benchmark must have a `evaluate` method that receives the program's `prediction` (output from the program's __call__ method) 
    # and the `label` (obtained using the `self.get_label` method) and return a dictionary of metrics. 
    def evaluate(self, prediction: Any, label: Any) -> dict:
        return super().evaluate(prediction, label)

class WorkFlow():
    def __init__(self,
                 summarizer,
                 aggregater,
                 reflector,
                 debater,
                 executer) -> None:
        
        self.summarizer = summarizer
        self.aggregater = aggregater
        self.reflector = reflector
        self.debater = debater
        self.executer = executer
        self.blocks = [self.summarizer, self.aggregater, self.reflector, self.debater, self.executer]
    
    def __call__(self, problem: str, **kwargs):
        entry_point = kwargs.get("entry_point", None)
        context = kwargs.get("context", None)
        # Step 1: 获取总结的上下文
        context = self.summarizer.execute(problem, context = context)
        
        # Step 2: 生成候选解决方案
        if self.debater.n > 0:
            # 需要辩论，生成多个候选方案
            candidate_solutions = self.aggregater.execute(problem, context=context)
        else:
            # 没有辩论，使用self consistency
            candidate_solution, _ = self.aggregater(problem, context=context)
            candidate_solutions = [candidate_solution]
        
        # Step 3: 对每个候选方案进行反思优化
        solutions = []
        for solution in candidate_solutions:
            if self.reflector.n > 0:
                refined_solution = self.reflector.execute(problem=problem, text=solution, context=context)
                solutions.append(refined_solution)
            else:
                refined_solution = self.executer.execute(problem=solution, context = context, entry_point = entry_point)
                solutions.append(solution)
        
        # Step 4: 通过辩论选择最佳答案
        final_answer = self.debater.execute(problem, solutions, context=context)
        
        return final_answer

    def save(self, path):
        params = {
            "summarizer": {
                "n": self.summarizer.n,
                "summarizer": self.summarizer.summarizer.prompt,
                "predictor": self.summarizer.predictor.prompt,
            },
            "aggregater":{
                "n": self.aggregater.n,
                "predictor": self.aggregater.predictor.prompt,
            },
            "reflector": {
                "n": self.reflector.n,
                "reflector": self.reflector.reflector.prompt,
                "refiner": self.reflector.refiner.prompt,
            },
            "debater": {
                "n": self.debater.n,
                "debater": self.debater.debater.prompt,
                "predictor": self.debater.predictor.prompt,
            },
            "executer": {
                "n": self.executer.n,
                "predictor": self.executer.predictor.prompt,
                "executer": self.executer.code_reflector.prompt,
            }
        }
        with open(path, "w") as f:
            json.dump(params, f)

def get_save_path(program):
    return f"examples/mass/{program}"

def mipro_optimize(registry, program, llm, save_path, dataset):
    optimizer = MiproOptimizer(
        registry = registry,
        program = program,
        optimizer_llm = llm,
        max_bootstrapped_demos= MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos = MAX_LABELED_DEMOS,
        num_threads = NUM_THREADS,
        eval_rounds= EVALUATION_ROUNDS,
        auto = AUTO,
        save_path = save_path
    )

    optimizer.optimize(dataset = dataset)

    return optimizer

def optimize_block(block, block_name, registry_tracks, benchmark, optimizer_llm, predictor_score, save_path_prefix):
    """
    统一的 block 优化函数
    
    Args:
        block: 要优化的 block 对象
        block_name: block 名称，用于保存路径
        registry_tracks: registry 跟踪配置列表，每个元素为 (track_path, input_names, output_names)
        benchmark: 基准测试数据集
        optimizer_llm: 优化器 LLM
        predictor_score: 基准 predictor 分数
        save_path_prefix: 保存路径前缀
    
    Returns:
        optimized_block: 优化后的 block，已设置 influence_score
    """
    # 创建 registry
    registry = MiproRegistry()
    for track_path, input_names, output_names in registry_tracks:
        registry.track(block, track_path, input_names=input_names, output_names=output_names)
    
    # 优化
    optimizer = mipro_optimize(
        registry=registry,
        program=block,
        llm=optimizer_llm,
        save_path=get_save_path(f"{save_path_prefix}/{block_name}"),
        dataset=benchmark
    )
    
    # 评估并计算影响分数
    score = optimizer.evaluate(dataset=benchmark, eval_mode="test")
    influence = score / predictor_score
    
    # 恢复最佳程序并设置影响分数
    optimized_block = optimizer.restore_best_program()
    optimized_block.influence_score = influence
    
    return optimized_block

def optimize_blocks_batch(block_configs, benchmark, optimizer_llm, optimized_predictor, executor_llm, predictor_score):
    """
    批量优化 blocks 的配置驱动函数
    
    Args:
        block_configs: block 配置列表，每个配置包含 block 的构造信息和 registry 配置
        benchmark: 基准测试数据集
        optimizer_llm: 优化器 LLM
        optimized_predictor: 已优化的 predictor
        executor_llm: 执行器 LLM
        predictor_score: 基准分数
        
    Returns:
        dict: 优化后的 blocks 字典
    """
    optimized_blocks = {}
    
    for config in block_configs:
        block_name = config['name']
        block_factory = config['factory']
        registry_tracks = config['tracks']
        
        # 创建 block
        if 'requires_llm' in config and config['requires_llm']:
            block = block_factory(predictor=optimized_predictor, llm=executor_llm)
        else:
            block = block_factory(predictor=optimized_predictor)
        
        # 优化 block
        optimized_block = optimize_block(
            block=block,
            block_name=block_name,
            registry_tracks=registry_tracks,
            benchmark=benchmark,
            optimizer_llm=optimizer_llm,
            predictor_score=predictor_score,
            save_path_prefix="mass"
        )
        
        optimized_blocks[block_name] = optimized_block
    
    return optimized_blocks

def main():
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    executor_llm = OpenAILLM(config=openai_config)
    optimizer_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
    optimizer_llm = OpenAILLM(config=optimizer_config)

    benchmark = MathSplits()
    
    # Step 0: Optimize Predictor 0
    predictor = Predictor(llm=executor_llm)
    predictor_registry = MiproRegistry()
    predictor_registry.track(predictor, "prompt", input_names=['problem', "context"], output_names=['answer'])
    optimized_predictor = mipro_optimize(
        registry=predictor_registry,
        program=predictor,
        llm=optimizer_llm,
        save_path=get_save_path("predictor"),
        dataset=benchmark
    )
    
    predictor_score = optimized_predictor.evaluate(dataset=benchmark, eval_mode="test")

    # Step 1: 配置驱动的批量优化
    block_configs = [
        {
            'name': 'summarizer',
            'factory': summarize,
            'requires_llm': True,
            'tracks': [
                ("summarizer.prompt", ['problem', 'context'], ['summary', 'reasoning', 'answer']),
                ("predictor.prompt", ['problem', 'context'], ['summary', 'reasoning', 'answer'])
            ]
        },
        {
            'name': 'aggregator', 
            'factory': aggregate,
            'requires_llm': False,
            'tracks': [
                ("predictor.prompt", ['problem', 'context'], ['answer'])
            ]
        },
        {
            'name': 'reflector',
            'factory': reflect,
            'requires_llm': True,
            'tracks': [
                ("predictor.prompt", ['problem', 'context'], ['answer']),
                ("reflector.prompt", ['problem', 'context'], ['answer']),
                ("refiner.prompt", ['problem', 'context'], ['answer'])
            ]
        },
        {
            'name': 'debater',
            'factory': debate,
            'requires_llm': True,
            'tracks': [
                ("debater.prompt", ['problem', 'context'], ['answer']),
                ("predictor.prompt", ['problem', 'context'], ['answer'])
            ]
        },
        {
            'name': 'executer',
            'factory': execute,
            'requires_llm': False,
            'tracks': [
                ("predictor.prompt", ['problem', 'entry_point', 'testcases'], ['answer']),
                ("executer.prompt", ['problem', 'entry_point', 'testcases'], ['answer'])
            ]
        }
    ]
    
    # 批量优化所有 blocks
    optimized_blocks = optimize_blocks_batch(
        block_configs=block_configs,
        benchmark=benchmark,
        optimizer_llm=optimizer_llm,
        optimized_predictor=optimized_predictor,
        executor_llm=executor_llm,
        predictor_score=predictor_score
    )

    return
    # 构建最终工作流
    block_workflow = WorkFlow(
        summarizer=optimized_blocks['summarizer'],
        aggregater=optimized_blocks['aggregator'],
        reflector=optimized_blocks['reflector'],
        debater=optimized_blocks['debater'],
        executer=optimized_predictor
    )


if __name__ == "__main__":
    main()