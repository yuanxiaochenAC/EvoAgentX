from evoagentx.config import Config
# from evoagentx.agents.agent import Agent
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.workflow.workflow_graph import WorkFlowGraph
from evoagentx.workflow.workflow import WorkFlow
from evoagentx.workflow.controller import WorkFlowController

# test LLM
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.models.litellm_model import LiteLLM


OPENAI_API_KEY = "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" # siwei's key
openai_config = OpenAILLMConfig(model = "gpt-4", openai_key = OPENAI_API_KEY, stream=True, output_response=True)
model = OpenAILLM(config=openai_config)

agent_manager = AgentManager()
wf_generator = WorkFlowGenerator(llm=model)

# generate workflow & agents
goal= "Create a web application that allows users to ask questions and get respond using ChatGPT with a key."
workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
workflow_graph.display()
workflow_graph.save_module("debug/workflow_demo.json")

workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file("debug/workflow_demo_v2.json")
agent_manager.add_agents_from_workflow(workflow_graph)

# execute workflow
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=model)
workflow.execute()

# variables = {"goal": workflow_graph.goal}
# for agent in agent_manager.agents:
#     action_name = agent.customize_action_name
#     result = agent.execute(
#         action_name=action_name, 
#         action_input_data=variables, 
#     ).content
#     variables.update(result.get_structured_data())