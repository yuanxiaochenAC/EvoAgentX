import os 
from dotenv import load_dotenv 
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager

load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM configuration
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",       
    openai_key=OPENAI_API_KEY, 
    stream=True, 
    output_response=True
)

# Initialize the language model
llm = OpenAILLM(config=openai_config)

goal = "Generate a python code for greedy snake game"
wf_generator = WorkFlowGenerator(llm=llm)
workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

# [optional] display workflow
workflow_graph.display()
# [optional] save workflow 
workflow_graph.save_module("debug/workflow_demo_4o_mini.json")
#[optional] load saved workflow 
workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file("debug/workflow_demo_4o_mini.json")

agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph)

workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
output = workflow.execute()
print(output)
