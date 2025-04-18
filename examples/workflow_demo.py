import os 
from dotenv import load_dotenv 
from evoagentx.models import OpenAILLMConfig, OpenAILLM, LiteLLMConfig, LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction

load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") 

def main():

    # LLM configuration
    openai_config = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=OPENAI_API_KEY, 
        stream=True, 
        output_response=True
    )

    # Initialize the language model
    llm = OpenAILLM(config=openai_config)

    goal = "Generate html code for the Tetris game that can be played in the browser."
    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    # [optional] display workflow
    workflow_graph.display()
    # [optional] save workflow 
    workflow_graph.save_module("debug/workflow_demo_4o_mini.json")
    #[optional] load saved workflow 
    # workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file("debug/workflow_demo_4o_mini.json")
    
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph)

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()
    # print(output)

    code_extractor = CodeExtraction()
    results = code_extractor.execute(
        llm=llm, 
        inputs={
            "code_string": output, 
            "target_directory": "debug/test_code_extraction",
            "project_name": "tetris_game"
        }
    )

    print(f"Extracted {len(results.extracted_files)} files:")
    for filename, path in results.extracted_files.items():
        print(f"  - {filename}: {path}")
    
    if results.main_file:
        print(f"\nMain file: {results.main_file}")
        file_type = os.path.splitext(results.main_file)[1].lower()
        if file_type == '.html':
            print(f"You can open this HTML file in a browser to play the Tetris game")
        elif file_type == '.py':
            print(f"You can run this Python file with 'python {results.main_file}'")
        elif file_type == '.java':
            print(f"You can compile this Java file with 'javac {results.main_file}'")
        else:
            print(f"This is the main entry point for your application")
    

if __name__ == "__main__":
    main()
