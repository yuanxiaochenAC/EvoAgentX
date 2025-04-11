import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Literal, Tuple, Optional, Callable
from ..benchmark.benchmark import Benchmark
from ..models.base_model import BaseLLM
from ..core.logging import logger
from ..models.model_utils import cost_manager


# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class AFlowEvaluator:

    """
    AFlow-specific evaluator
    """

    def __init__( self, llm: Optional[BaseLLM] = None):
        self.llm = llm 
    
    def _configure_graph(self, graph, benchmark):
        return graph(name=benchmark.name, llm_config=self.llm.config, benchmark=benchmark)
    
    async def graph_evaluate_async(self, benchmark: Benchmark, graph: Callable, is_test: bool = False, max_concurrent_tasks: int = 20) -> Tuple[float, float, float]:
        
        configured_graph = self._configure_graph(graph=graph, benchmark=benchmark)

        # Get evaluation data
        data = benchmark.get_test_data() if is_test else benchmark.get_dev_data()
        if not data:
            logger.warning("No data to evaluate. Returning zeros.")
            return (0.0, 0.0, 0.0)
        
        # get total cost before evaluation
        cost_before = cost_manager.get_total_cost()
        
        # Create a shared semaphore
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        async def evaluate_with_semaphore(example):
            async with semaphore:
                try:
                    return await benchmark.evaluate_async(configured_graph, example)
                except Exception as e:
                    logger.warning(f"Evaluation failed: {str(e)}")
                    return None
        
        # Create tasks for concurrent execution with semaphore
        tasks = [evaluate_with_semaphore(example) for example in data]

        # Wait for all tasks to complete
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Evaluating {benchmark.name} problems",
            total=len(data)
        )

        # Replace failed evaluations (None results) with 0
        valid_results = [0.0 if r is None else r for r in results]

        # get total cost after evaluation
        total_cost = cost_manager.get_total_cost() - cost_before
        avg_cost = total_cost / len(data)

        if not valid_results:
            logger.warning("No valid results. Returning zeros.")
            avg_metrics = 0.0
        else:
            avg_metrics = sum(valid_results) / len(valid_results)
        
        return avg_metrics, avg_cost, total_cost
        
        

    