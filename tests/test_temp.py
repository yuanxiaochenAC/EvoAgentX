import pytest
import pytest_asyncio
import uuid
from urllib.parse import urljoin
import httpx
import os
import sys
import io
from unittest.mock import patch
from evoagentx.agents import Agent
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.app.services import AgentService
from evoagentx.app.db import Agent as AppAgent
from evoagentx.app.config import settings
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.prompts.bolt_prompt_system import BOLT_PROMPT
from evoagentx.actions.agent_generation import AgentGeneration
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.prompts.requirement_collection import REQUIREMENT_COLLECTION_PROMPT
from evoagentx.workflow.workflow import WorkFlow
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.workflow.workflow_graph import WorkFlowGraph, WorkFlowNode, Parameter, WorkFlowEdge
from evoagentx.core.base_config import Parameter
from evoagentx.actions.action import Action, ActionInput, ActionOutput
from evoagentx.core.message import Message, MessageType


# API KEY for OpenAI models
OPENAI_API_KEY="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"


# Custom action for question answering
class QAInput(ActionInput):
    question: str = ""

class QAOutput(ActionOutput):
    answer: str = ""
    
class SimpleQAAction(Action):
    """A simple action for question answering."""
    
    def __init__(self, name, **kwargs):
        description = kwargs.pop("description", "An action that answers questions using an LLM")
        super().__init__(name=name, description=description, inputs_format=QAInput, outputs_format=QAOutput, **kwargs)
    
    def execute(self, llm, inputs=None, **kwargs):
        """Execute the QA action with the given inputs."""
        # Extract the question from inputs
        question = inputs.get("question", "")
        if not question:
            return None
        
        # Generate an answer using the LLM
        prompt = f"Question: {question}\n\nProvide a clear, accurate, and concise answer to this question."
        response = llm.generate(prompt=prompt)
        
        # Format the output
        output = self.outputs_format()
        output.answer = response.content
        
        # Return a message
        return Message(
            content={"answer": response.content},
            msg_type=MessageType.RESPONSE,
            action=self.name
        )


# We'll patch the workflow's execute method to intercept the answer
class WorkFlowWrapper(WorkFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_answer = None
        
    def execute_task(self, task, **kwargs):
        # Intercept output before passing to parent method
        result = super().execute_task(task, **kwargs)
        
        return result


@pytest.mark.asyncio
async def test_local_simple_qa_workflow():
    """
    Demonstrates creating and executing a simple QA workflow locally without using the API.
    
    This test:
    1. Creates an agent with QA capabilities
    2. Adds a QA action to the agent
    3. Creates a workflow graph with a single task
    4. Executes the workflow locally
    5. Verifies the output
    """
    print("\n=== Testing Simple QA Workflow Locally ===")
    
    # Step 1: Create the LLM
    print("Creating LLM...")
    llm_config = OpenAILLMConfig(
        model="gpt-3.5-turbo", 
        openai_key=OPENAI_API_KEY,
        temperature=0.3,
        max_tokens=500,
        top_p=0.9
    )
    llm = OpenAILLM(config=llm_config)
    
    # Step 2: Create an agent manager and add a QA agent
    print("Creating agent manager and QA agent...")
    agent_manager = AgentManager()
    agent_manager.init_module()
    
    # Create QA agent
    agent_manager.add_agent({
        "name": "QAAgent",
        "description": "An agent that can answer questions based on provided content",
        "prompt": "You are a helpful assistant that provides accurate, concise answers to questions.",
        "llm_config": llm_config
    })
    
    # Add QA action to the agent
    qa_agent = agent_manager.get_agent("QAAgent")
    qa_action = SimpleQAAction(name="answer_question")
    qa_agent.add_action(qa_action)
    
    # Step 3: Create a workflow graph with a single task
    print("Creating workflow graph...")
    
    # Define workflow inputs and outputs
    workflow_inputs = [Parameter(name="question", type="str", description="The question to be answered")]
    workflow_outputs = [Parameter(name="answer", type="str", description="The answer to the question")]
    
    # Create workflow graph
    workflow_graph = WorkFlowGraph(
        name="SimpleQAWorkflow",
        description="A simple workflow that answers questions",
        goal="Answer the user's question accurately and concisely",
        inputs=workflow_inputs,
        outputs=workflow_outputs
    )
    
    # Create a workflow node (task)
    qa_task = WorkFlowNode(
        name="QuestionAnswering",
        description="Answer the provided question",
        inputs=[Parameter(name="question", type="str", description="The question to be answered")],
        outputs=[Parameter(name="answer", type="str", description="The answer to the question")],
        agents=["QAAgent"]  # Specify which agent will execute this task
    )
    
    # Add node to graph
    workflow_graph.add_node(qa_task)
    
    # In this simple workflow, we don't need edges since it's a single node
    # The workflow will automatically connect inputs to the node
    
    # Step 4: Create workflow using our wrapper class
    print("Creating and executing workflow...")
    workflow = WorkFlowWrapper(
        graph=workflow_graph,
        llm=llm,
        agent_manager=agent_manager
    )
    
    # Step 5: Execute workflow
    question = "What are the key benefits of using multi-agent workflows compared to single-agent approaches?"
    workflow_input = {"question": question}
    
    # Capture stdout to extract the answer
    
    try:
        result = workflow.execute(inputs=workflow_input)
        
        # Try to get answer from our wrapper's captured_answer
        result = workflow.captured_answer
               
        print(f"\nQuestion: {question}")
        print(f"Answer: {result}\n")
        
        # Verify that we got a valid result
        assert isinstance(result, str)
        assert len(result) > 0
        print("Workflow execution completed successfully!")
    except Exception as e:
        print(f"Error during workflow execution: {str(e)}")
        raise(e)
    
    assert False


# from evoagentx.models import OpenAILLMConfig, OpenAILLM
# from evoagentx.agents import AgentManager
# from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow

# # set output_response=True to see LLM outputs 
# openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=False)
# model = OpenAILLM(config=openai_config)

# agent_manager = AgentManager()
# wf_generator = WorkFlowGenerator(llm=model)

# # generate workflow & agents
# workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal="Generate a python code for greedy snake game")

# # [optional] display workflow
# workflow_graph.display()
# # [optional] save workflow 
# workflow_graph.save_module("debug/workflow_demo.json")
# #[optional] load saved workflow 
# workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file("debug/workflow_demo.json")

# agent_manager.add_agents_from_workflow(workflow_graph)
# # execute workflow
# workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=model)
# output = workflow.execute()
# print(output)

# @pytest.mark.asyncio
# async def test_local_multi_step_workflow():
#     """
#     Demonstrates creating and executing a multi-step workflow locally without using the API.
    
#     This test creates a workflow with three steps:
#     1. Data preparation: Extract relevant information from raw data
#     2. Analysis: Analyze the extracted data
#     3. Reporting: Generate a summary report of the analysis
    
#     This demonstrates how to create a more complex workflow with multiple agents
#     and data dependencies between steps.
#     """
#     print("\n=== Testing Multi-Step Workflow Locally ===")
    
#     # Step 1: Create the LLM
#     print("Creating LLM...")
#     llm_config = OpenAILLMConfig(
#         model="gpt-3.5-turbo", 
#         openai_key=OPENAI_API_KEY,
#         temperature=0.3,
#         max_tokens=500,
#         top_p=0.9
#     )
#     llm = OpenAILLM(config=llm_config)
    
#     # Step 2: Create custom actions for each step in the workflow
    
#     # Data Preparation Action
#     class DataPrepAction(Action):
#         """Action for preparing data for analysis."""
        
#         def __init__(self, name, **kwargs):
#             inputs_format = ActionInput
#             class DataPrepOutputFormat(ActionOutput):
#                 extracted_data: str = ""
            
#             super().__init__(
#                 name=name, 
#                 description="Extracts and prepares data for analysis", 
#                 inputs_format=inputs_format,
#                 outputs_format=DataPrepOutputFormat,
#                 **kwargs
#             )
        
#         def execute(self, llm, inputs=None, **kwargs):
#             raw_data = inputs.get("raw_data", "")
#             prompt = f"""
#             Given the following raw data:
#             {raw_data}
            
#             Extract the key information and prepare it for analysis. Format your response as a structured summary.
#             """
#             response = llm.generate(prompt=prompt)
            
#             return Message(
#                 content={"extracted_data": response.content},
#                 msg_type=MessageType.RESPONSE,
#                 action=self.name
#             )
    
#     # Analysis Action
#     class AnalysisAction(Action):
#         """Action for analyzing prepared data."""
        
#         def __init__(self, name, **kwargs):
#             class AnalysisInputFormat(ActionInput):
#                 extracted_data: str = ""
                
#             class AnalysisOutputFormat(ActionOutput):
#                 analysis_results: str = ""
            
#             super().__init__(
#                 name=name, 
#                 description="Analyzes prepared data to generate insights", 
#                 inputs_format=AnalysisInputFormat,
#                 outputs_format=AnalysisOutputFormat,
#                 **kwargs
#             )
        
#         def execute(self, llm, inputs=None, **kwargs):
#             extracted_data = inputs.get("extracted_data", "")
#             prompt = f"""
#             Given the following prepared data:
#             {extracted_data}
            
#             Analyze this data and identify key trends, patterns, and insights. Provide a detailed analysis.
#             """
#             response = llm.generate(prompt=prompt)
            
#             return Message(
#                 content={"analysis_results": response.content},
#                 msg_type=MessageType.RESPONSE,
#                 action=self.name
#             )
    
#     # Reporting Action
#     class ReportingAction(Action):
#         """Action for generating a report based on analysis results."""
        
#         def __init__(self, name, **kwargs):
#             class ReportingInputFormat(ActionInput):
#                 analysis_results: str = ""
                
#             class ReportingOutputFormat(ActionOutput):
#                 final_report: str = ""
            
#             super().__init__(
#                 name=name, 
#                 description="Generates a comprehensive report based on analysis results", 
#                 inputs_format=ReportingInputFormat,
#                 outputs_format=ReportingOutputFormat,
#                 **kwargs
#             )
        
#         def execute(self, llm, inputs=None, **kwargs):
#             analysis_results = inputs.get("analysis_results", "")
#             prompt = f"""
#             Based on the following analysis results:
#             {analysis_results}
            
#             Generate a comprehensive, well-structured final report that summarizes the key findings and provides actionable recommendations.
#             """
#             response = llm.generate(prompt=prompt)
            
#             return Message(
#                 content={"final_report": response.content},
#                 msg_type=MessageType.RESPONSE,
#                 action=self.name
#             )
    
#     # Step 3: Create an agent manager and add agents for each step
#     print("Creating agent manager and agents...")
#     agent_manager = AgentManager()
#     agent_manager.init_module()
    
#     # Create agents for each step
#     agent_manager.add_agent({
#         "name": "DataPrepAgent",
#         "description": "An agent that prepares and extracts data from raw sources",
#         "prompt": "You are a data preparation specialist that can extract structured information from raw data.",
#         "llm_config": llm_config
#     })
    
#     agent_manager.add_agent({
#         "name": "AnalysisAgent",
#         "description": "An agent that analyzes prepared data to generate insights",
#         "prompt": "You are a data analyst that can identify patterns, trends, and insights from prepared data.",
#         "llm_config": llm_config
#     })
    
#     agent_manager.add_agent({
#         "name": "ReportingAgent",
#         "description": "An agent that generates comprehensive reports based on analysis",
#         "prompt": "You are a reporting specialist that can create clear, concise, and informative reports.",
#         "llm_config": llm_config
#     })
    
#     # Add actions to agents
#     data_prep_agent = agent_manager.get_agent("DataPrepAgent")
#     data_prep_agent.add_action(DataPrepAction(name="prepare_data"))
    
#     analysis_agent = agent_manager.get_agent("AnalysisAgent")
#     analysis_agent.add_action(AnalysisAction(name="analyze_data"))
    
#     reporting_agent = agent_manager.get_agent("ReportingAgent")
#     reporting_agent.add_action(ReportingAction(name="generate_report"))
    
#     # Step 4: Create a workflow graph with multiple steps
#     print("Creating workflow graph...")
    
#     # Define workflow inputs and outputs
#     workflow_inputs = [Parameter(name="raw_data", type="str", description="The raw data to be processed")]
#     workflow_outputs = [Parameter(name="final_report", type="str", description="The final report with analysis and recommendations")]
    
#     # Create workflow graph
#     workflow_graph = WorkFlowGraph(
#         name="DataAnalysisWorkflow",
#         description="A workflow that processes raw data, analyzes it, and generates a report",
#         goal="Generate insights and recommendations from raw data",
#         inputs=workflow_inputs,
#         outputs=workflow_outputs
#     )
    
#     # Create workflow nodes (tasks)
#     prep_task = WorkFlowNode(
#         name="DataPreparation",
#         description="Extract and prepare data from raw sources",
#         inputs=[Parameter(name="raw_data", type="str", description="The raw data to be processed")],
#         outputs=[Parameter(name="extracted_data", type="str", description="Extracted and prepared data")],
#         agents=["DataPrepAgent"]
#     )
    
#     analysis_task = WorkFlowNode(
#         name="DataAnalysis",
#         description="Analyze the prepared data to generate insights",
#         inputs=[Parameter(name="extracted_data", type="str", description="Extracted and prepared data")],
#         outputs=[Parameter(name="analysis_results", type="str", description="Analysis results and insights")],
#         agents=["AnalysisAgent"]
#     )
    
#     reporting_task = WorkFlowNode(
#         name="ReportGeneration",
#         description="Generate a comprehensive report based on analysis",
#         inputs=[Parameter(name="analysis_results", type="str", description="Analysis results and insights")],
#         outputs=[Parameter(name="final_report", type="str", description="The final report with analysis and recommendations")],
#         agents=["ReportingAgent"]
#     )
    
#     # Add nodes to graph
#     workflow_graph.add_node(prep_task)
#     workflow_graph.add_node(analysis_task)
#     workflow_graph.add_node(reporting_task)
    
#     # Connect nodes using WorkFlowEdge
#     # Connect first task to second task
#     edge1 = WorkFlowEdge(edge_tuple=(prep_task.name, analysis_task.name))
#     workflow_graph.add_edge(edge1)
    
#     # Connect second task to third task
#     edge2 = WorkFlowEdge(edge_tuple=(analysis_task.name, reporting_task.name))
#     workflow_graph.add_edge(edge2)
    
#     # Step 5: Create workflow
#     print("Creating and executing workflow...")
#     workflow = WorkFlow(
#         graph=workflow_graph,
#         llm=llm,
#         agent_manager=agent_manager
#     )
    
#     # Step 6: Execute workflow
#     raw_data = """
#     Sales Data 2023:
#     Q1: 
#     - Product A: $125,000
#     - Product B: $85,000
#     - Product C: $42,000
    
#     Q2:
#     - Product A: $142,000
#     - Product B: $78,000
#     - Product C: $53,000
    
#     Q3:
#     - Product A: $138,000
#     - Product B: $92,000
#     - Product C: $61,000
    
#     Q4:
#     - Product A: $156,000
#     - Product B: $104,000
#     - Product C: $72,000
    
#     Customer retention rate: 78%
#     New customer acquisition: 156
#     Customer churn rate: 12%
#     """
    
#     workflow_input = {"raw_data": raw_data}
    
#     try:
#         result = workflow.execute(inputs=workflow_input)
#         print("\nFinal Report:")
#         print(result)
        
#         # Verify that we got a valid result
#         assert isinstance(result, str)
#         assert len(result) > 0
#         print("Multi-step workflow execution completed successfully!")
#     except Exception as e:
#         print(f"Error during workflow execution: {str(e)}")
#         raise
   

   