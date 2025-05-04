from evoagentx.optimizers.mipro_optimizer import MiproOptimizer
from evoagentx.models import OpenAILLMConfig, OpenAILLM, LiteLLM
from evoagentx.benchmark.humaneval import HumanEval
from evoagentx.utils.mipro_utils.settings import settings
from evoagentx.workflow.workflow_graph import WorkFlowGraph

def main():
    graph = WorkFlowGraph.from_file("examples/output/saved_sequential_workflow.json")
    print(type(graph))
    print(type(graph.deepcopy()))
if __name__ == "__main__":
    main()
