import os 
from dotenv import load_dotenv 
from evoagentx.models import OpenAILLMConfig, OpenAILLM, LiteLLMConfig, LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.workflow.workflow_manager import WorkFlowManager
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification 
from evoagentx.core.module_utils import extract_code_blocks
import asyncio
load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") 

pdf_path = "examples/test_pdf.pdf"
mcp_config_path = "examples/mcp.config"

async def main():

    # LLM configuration
    openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000)
    # Initialize the language model
    llm = OpenAILLM(config=openai_config)

    goal = f"Read and analyze the PDF file at {pdf_path}, then find real job opportunities by search the website."
    target_directory = "examples/output/pdf_demo"
    
    # wf_generator = WorkFlowGenerator(llm=llm, mcp_config_path=mcp_config_path)
    # workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
    # # [optional] display workflow
    # workflow_graph.display()
    # # [optional] save workflow 
    # workflow_graph.save_module(f"{target_directory}/workflow_demo_4o_mini.json")
    
    
    #[optional] load saved workflow 
    workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(f"{target_directory}/workflow_demo_4o_mini.json")
    print(workflow_graph)

    agent_manager = AgentManager()
    await agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    workflow.init_module()
    output = await workflow.execute()
    from pdb import set_trace; set_trace()
    
    print(output)
    
    # verfiy the code
    

if __name__ == "__main__":
    asyncio.run(main())