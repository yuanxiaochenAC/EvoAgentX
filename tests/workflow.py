from evoagentx.config import Config
# from evoagentx.agents.agent import Agent
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.workflow.workflow_graph import WorkFlowGraph, WorkFlowNodeState, WorkFlowEdge, WorkFlowNode
from evoagentx.workflow.workflow import WorkFlow
from evoagentx.workflow.controller import WorkFlowController

# test LLM
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.models.litellm_model import LiteLLM


OPENAI_API_KEY = "sk-" # OpenAI's KEY 
ANTHROPIC_API_KEY = "sk-" 
SILICONFLOW_KEY = "sk-"

openai_config = LiteLLMConfig(model = "gpt-4o-mini", openai_key = OPENAI_API_KEY, stream=True)
model = LiteLLM(config=openai_config)

workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file("debug/workflow_demo.json")
# workflow_graph.display()

# test_loop 
workflow_graph.add_edge(WorkFlowEdge(("test_game_functionality", "design_game_structure")))
workflow_graph.add_edge(WorkFlowEdge(("optimize_game_performance", "design_game_structure")))

node = WorkFlowNode.from_dict(
    {
        "name": "test", 
        "description": "desc", 
        "inputs": [{"name": "input", "type": "xxx", "description": "xxx"}], 
        "outputs": [{"name": "output", "type": "xxx", "description": "xxx"}], 
    }
)
workflow_graph.add_node(node)
node = WorkFlowNode.from_dict(
    {
        "name": "test2", 
        "description": "desc", 
        "inputs": [{"name": "input", "type": "xxx", "description": "xxx"}], 
        "outputs": [{"name": "output", "type": "xxx", "description": "xxx"}], 
    }
)
workflow_graph.add_node(node)

node = WorkFlowNode.from_dict(
    {
        "name": "test3", 
        "description": "desc", 
        "inputs": [{"name": "input", "type": "xxx", "description": "xxx"}], 
        "outputs": [{"name": "output", "type": "xxx", "description": "xxx"}], 
    }
)
workflow_graph.add_node(node)
node = WorkFlowNode.from_dict(
    {
        "name": "test4", 
        "description": "desc", 
        "inputs": [{"name": "input", "type": "xxx", "description": "xxx"}], 
        "outputs": [{"name": "output", "type": "xxx", "description": "xxx"}], 
    }
)
workflow_graph.add_node(node)

workflow_graph.add_edge(WorkFlowEdge(("test", "test2")))
workflow_graph.add_edge(WorkFlowEdge(("test2", "implement_game_logic")))
workflow_graph.add_edge(WorkFlowEdge(("optimize_game_performance", "test2")))

workflow_graph.add_edge(WorkFlowEdge(("test3", "test4")))
workflow_graph.add_edge(WorkFlowEdge(("test4", "implement_game_logic")))
workflow_graph.add_edge(WorkFlowEdge(("optimize_game_performance", "test4")))

# workflow_graph.get_next_candidate_nodes()

workflow_graph.completed("define_game_requirements")
workflow_graph.completed("test")
workflow_graph.completed("test3")

workflow_graph.completed("design_game_structure")
workflow_graph.completed("test2")
workflow_graph.completed("test4")
workflow_graph.completed("implement_game_logic")
workflow_graph.completed("test_game_functionality")

workflow_graph.display()
# workflow_graph.set_node_status("optimize_game_performance", WorkFlowNodeState.COMPLETED)
# workflow_graph.set_node_status("launch_game", WorkFlowNodeState.COMPLETED)

workflow_graph.get_next_candidate_nodes()
workflow_graph.step("test_game_functionality", "optimize_game_performance")
workflow_graph.completed("optimize_game_performance")
workflow_graph.get_next_candidate_nodes()
workflow_graph.step("optimize_game_performance", "design_game_structure")
from pdb import set_trace; set_trace()

workflow_graph.completed("design_game_structure")
workflow_graph.completed("implement_game_logic")
workflow_graph.completed("test_game_functionality")
workflow_graph.step("test_game_functionality", "test4")
from pdb import set_trace; set_trace()
# variables = {"goal": workflow_graph.goal} 
# for agent in agent_manager.agents:
#     action_name = agent.customize_action_name
#     result = agent.execute(
#         action_name=action_name, 
#         action_input_data=variables, 
#     ).content
#     from pdb import set_trace; set_trace()
#     variables.update(result.get_structured_data())