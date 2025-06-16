import asyncio
from datetime import datetime
from typing import Dict, Any

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.models.model_utils import create_llm_instance

from evoagentx.agents.agent_manager import AgentManager
from evoagentx.tools.mcp import MCPToolkit

import uuid

from .task_manager import (
    store_task_result,
    create_stream_task,
    update_stream_task,
    complete_stream_task,
    send_to_client,
    remove_task_from_client
)
from .models import ProcessResponse

sudo_workflow = WorkFlow.from_file("examples/output/jobs/jobs_demo_4o_mini.json")
# sudo_workflow = None

def create_llm_config(llm_config_dict: Dict[str, Any]) -> LLMConfig:
    """
    Convert a dictionary to the appropriate LLM config object based on the model type.
    """
    model = llm_config_dict.get("model", "").lower()
    
    # Determine the appropriate config class based on the model
    if "gpt" in model or "openai" in model:
        return OpenAILLMConfig(**llm_config_dict)
    else:
        # Default to OpenAI config if we can't determine the type
        # You might want to add more specific logic here
        try:
            return OpenAILLMConfig(**llm_config_dict)
        except Exception:
            # If OpenAI config fails, try the base LLMConfig
            return LLMConfig(**llm_config_dict)

async def process_task(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for the actual processing logic.
    This is where you'll implement your specific processing functionality.
    """
    # Simulate some processing time
    await asyncio.sleep(2)
    
    # Example processing - replace this with your actual logic
    return {
        "processed": True,
        "input_parameters": config,
        "sample_output": "This is a sample result"
    }

async def handle_process_request(config: Dict[str, Any]) -> ProcessResponse:
    """Handle a processing request and return a response"""
    task_id = str(uuid.uuid4())
    
    # Process the task
    result = await process_task(config)
    
    # Create response
    response = ProcessResponse(
        task_id=task_id,
        status="completed",
        result=result
    )
    
    # Store the result
    store_task_result(task_id, response)
    
    return response

async def process_stream_task(task_id: str, config: Dict[str, Any]):
    """
    Process a streaming task and generate updates.
    """
    total_steps = 5
    for step in range(total_steps):
        # Simulate processing time for each step
        await asyncio.sleep(1)
        
        # Update progress
        progress = {
            "step": step + 1,
            "total_steps": total_steps,
            "timestamp": datetime.now().isoformat(),
            "status": "processing",
            "progress": ((step + 1) / total_steps) * 100,
            "current_state": f"Processing step {step + 1}/{total_steps}"
        }
        
        update_stream_task(task_id, progress)
    
    # Final result
    final_result = {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "result": {
            "processed": True,
            "input_parameters": config,
            "final_output": "Streaming task completed successfully"
        }
    }
    
    update_stream_task(task_id, final_result)
    complete_stream_task(task_id)

async def start_streaming_task(config: Dict[str, Any]) -> Dict[str, Any]:
    """Start a new streaming task"""
    task_id = str(uuid.uuid4())
    
    # Initialize the stream task
    create_stream_task(task_id, config)
    
    # Determine task type and start appropriate processing
    task_type = config.get("task_type", "default")
    
    if task_type == "workflow_generation":
        # Start workflow generation in the background
        asyncio.create_task(process_workflow_generation_task(task_id, config["parameters"]))
    elif task_type == "workflow_execution":
        # Start workflow execution in the background
        asyncio.create_task(process_workflow_execution_task(task_id, config["parameters"]))
    else:
        # Default processing task
        asyncio.create_task(process_stream_task(task_id, config["parameters"]))
    
    return {
        "task_id": task_id,
        "status": "started",
        "stream_url": f"/stream/{task_id}",
        "task_type": task_type
    } 

async def process_workflow_generation_task(task_id: str, config: Dict[str, Any]):
    """
    Process workflow generation as a streaming task.
    """
    try:
        goal = config.get("goal")
        llm_config_dict = config.get("llm_config")
        mcp_config = config.get("mcp_config", {})
        
        if not goal or not llm_config_dict:
            error_update = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": "Missing required parameters: goal or llm_config"
            }
            update_stream_task(task_id, error_update)
            complete_stream_task(task_id)
            return
        
        # Generate the workflow
        workflow_graph = await generate_workflow_from_goal(goal, llm_config_dict, mcp_config)
        
        if workflow_graph is None:
            error_update = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": "Failed to generate workflow"
            }
            update_stream_task(task_id, error_update)
            complete_stream_task(task_id)
            return
        
        # Convert workflow_graph to serializable format
        try:
            if hasattr(workflow_graph, 'get_config'):
                workflow_dict = workflow_graph.get_config()
            elif hasattr(workflow_graph, 'get_workflow_description'):
                workflow_dict = {
                    "goal": workflow_graph.goal,
                    "description": workflow_graph.get_workflow_description()
                }
            else:
                workflow_dict = str(workflow_graph)
        except Exception as e:
            workflow_dict = f"Workflow generated successfully (serialization error: {str(e)})"
        
        final_result = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "result": {
                "workflow_graph": workflow_dict,
                "goal": goal,
                "message": "Workflow generated successfully"
            }
        }
        
        update_stream_task(task_id, final_result)
        complete_stream_task(task_id)
        
    except Exception as e:
        error_update = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Unexpected error during workflow generation: {str(e)}"
        }
        update_stream_task(task_id, error_update)
        complete_stream_task(task_id)

# New client-aware processing function
async def process_workflow_generation_for_client(client_id: str, task_id: str, config: Dict[str, Any]):
    """
    Process workflow generation and send updates to a specific client.
    """
    try:
        goal = config.get("goal")
        llm_config_dict = config.get("llm_config")
        mcp_config = config.get("mcp_config", {})
        
        if not goal or not llm_config_dict:
            send_to_client(client_id, {
                "event_type": "task_error",
                "task_id": task_id,
                "error": "Missing required parameters: goal or llm_config"
            })
            remove_task_from_client(client_id, task_id)
            return
        
        # Send processing start notification
        send_to_client(client_id, {
            "event_type": "task_processing",
            "task_id": task_id,
            "message": f"Starting workflow generation for: {goal[:50]}..."
        })
        
        # Generate the workflow
        workflow_graph = await generate_workflow_from_goal(goal, llm_config_dict, mcp_config)
        
        if workflow_graph is None:
            send_to_client(client_id, {
                "event_type": "task_error",
                "task_id": task_id,
                "error": "Failed to generate workflow"
            })
            remove_task_from_client(client_id, task_id)
            return
        
        # Convert workflow_graph to serializable format
        try:
            if hasattr(workflow_graph, 'get_config'):
                workflow_dict = workflow_graph.get_config()
            elif hasattr(workflow_graph, 'get_workflow_description'):
                workflow_dict = {
                    "goal": workflow_graph.goal,
                    "description": workflow_graph.get_workflow_description()
                }
            else:
                workflow_dict = str(workflow_graph)
        except Exception as e:
            workflow_dict = f"Workflow generated successfully (serialization error: {str(e)})"
        
        # Send completion notification to client
        send_to_client(client_id, {
            "event_type": "task_completed",
            "task_id": task_id,
            "result": {
                "workflow_graph": workflow_dict,
                "goal": goal,
                "message": "Workflow generated successfully"
            }
        })
        
        remove_task_from_client(client_id, task_id)
        
    except Exception as e:
        send_to_client(client_id, {
            "event_type": "task_error",
            "task_id": task_id,
            "error": f"Unexpected error during workflow generation: {str(e)}"
        })
        remove_task_from_client(client_id, task_id)

async def generate_workflow_from_goal(goal: str, llm_config_dict: Dict[str, Any], mcp_config: dict = None) -> str:
    """
    Generate a workflow from a goal.
    """
    
    if sudo_workflow:
        return sudo_workflow
    
    try:
        # Convert dictionary to appropriate LLM config object and create LLM instance
        llm_config = create_llm_config(llm_config_dict)
        llm = create_llm_instance(llm_config)
        
        if mcp_config:
            tools = MCPToolkit(config=mcp_config)
        else:
            tools = []
    except Exception as e:
        print(f"Error initializing components: {e}")
        return None
    
    workflow_generator = WorkFlowGenerator(llm=llm, tools=tools)
    
    # Generate the workflow
    workflow_graph: WorkFlowGraph = workflow_generator.generate_workflow(goal=goal)
    return workflow_graph

async def execute_workflow_from_config(workflow: Dict[str, Any], llm_config_dict: Dict[str, Any], mcp_config: dict = None) -> Dict[str, Any]:
    """
    Execute a workflow with the given configuration.
    
    Args:
        workflow: The workflow definition/configuration to execute
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        
    Returns:
        Dict containing execution results and status
        
    """
    try:
        llm_config = create_llm_config(llm_config_dict)
        llm = create_llm_instance(llm_config)
        workflow_graph: WorkFlowGraph = WorkFlowGraph.from_dict(workflow)
        if mcp_config:
            mcp_toolkit = MCPToolkit(config=mcp_config)
            tools = mcp_toolkit.get_tools()
        else:
            tools = []
        
        agent_manager = AgentManager(tools=tools)
        agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)
        # from pdb import set_trace; set_trace()

        workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
        workflow.init_module()
        output = await workflow.async_execute()
        
        return {
            "status": "completed",
            "message": output,
            "workflow_received": bool(workflow),
            "llm_config_received": bool(llm_config_dict),
            "mcp_config_received": bool(mcp_config)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"In the execution process, got error:\n{e}",
            "workflow_received": bool(workflow),
            "llm_config_received": bool(llm_config_dict),
            "mcp_config_received": bool(mcp_config)
        }

async def process_workflow_execution_task(task_id: str, config: Dict[str, Any]):
    """
    Process workflow execution as a streaming task.
    Simple and reusable following the same pattern as workflow generation.
    """
    try:
        workflow_dict = config.get("workflow")
        llm_config_dict = config.get("llm_config")
        mcp_config = config.get("mcp_config", {})
        
        if not workflow_dict or not llm_config_dict:
            error_update = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": "Missing required parameters: workflow or llm_config"
            }
            update_stream_task(task_id, error_update)
            complete_stream_task(task_id)
            return
        
        # Send progress update before execution (as requested)
        pre_execution_update = {
            "status": "executing",
            "timestamp": datetime.now().isoformat(),
            "message": "Starting workflow execution...",
            "progress": 50
        }
        update_stream_task(task_id, pre_execution_update)
        
        # Execute the workflow (black box execution)
        execution_result = await execute_workflow_from_config(workflow_dict, llm_config_dict, mcp_config)
        
        if execution_result is None:
            error_update = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": "Failed to execute workflow"
            }
            update_stream_task(task_id, error_update)
            complete_stream_task(task_id)
            return
        
        # Send final result
        final_result = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "progress": 100,
            "result": {
                "execution_result": execution_result,
                "message": "Workflow execution completed"
            }
        }
        
        update_stream_task(task_id, final_result)
        complete_stream_task(task_id)
        
    except Exception as e:
        error_update = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": f"Unexpected error during workflow execution: {str(e)}"
        }
        update_stream_task(task_id, error_update)
        complete_stream_task(task_id)
