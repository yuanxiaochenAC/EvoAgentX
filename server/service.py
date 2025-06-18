import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.models import LLMConfig
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.core.module_utils import parse_json_from_text

from .prompts import WORKFLOW_GENERATION_PROMPT, TASK_INFO_PROMPT_SUDO, CONNECTION_INSTRUCTION_PROMPT

from .task_manager import (
    store_task_result,
    create_stream_task,
    update_stream_task,
    complete_stream_task,
    send_to_client,
    remove_task_from_client
)
from .models import ProcessResponse

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
default_llm_config = {
    "model": "gpt-4o-mini",
    "openai_key": OPENAI_API_KEY,
    "stream": True,
    "output_response": True,
    "max_tokens": 16000
}
TUNNEL_INFO_PATH = "./server/tunnel_info.json"
# sudo_workflow = WorkFlow.from_file("examples/output/jobs/jobs_demo_4o_mini.json")
sudo_workflow = None
# sudo_execution_result = "Sudo execution result for the given workflow."
sudo_execution_result = None


# In-memory project database
project_info: Dict[str, Dict[str, Any]] = {}

def read_tunnel_info():
    """Read tunnel information from JSON file"""
    try:
        if os.path.exists(TUNNEL_INFO_PATH):
            with open(TUNNEL_INFO_PATH, "r") as f:
                return json.load(f)
        return None
    except Exception:
        return None

def create_workflow_info(config: Dict[str, Any], execution_result: Dict[str, Any]) -> str:
    """Create comprehensive workflow info string including public URL and other details"""
    tunnel_info = read_tunnel_info()
    
    # Extract key information
    public_url = tunnel_info.get("public_url", "Not available") if tunnel_info else "Not available"
    local_url = tunnel_info.get("local_url", "Not available") if tunnel_info else "Not available"
    
    workflow_dict = config.get("workflow", {})
    
    # Build comprehensive workflow info string
    workflow_info = f"""
=== WORKFLOW EXECUTION INFORMATION ===
Timestamp: {datetime.now().isoformat()}

=== SERVER ACCESS INFORMATION ===
Public URL: {public_url}
Local URL: {local_url}

=== WORKFLOW CONFIGURATION ===
Workflow Status: {execution_result.get('status', 'Unknown')}
LLM Configuration: {config.get('llm_config', {}).get('model', 'Unknown')}
MCP Configuration: {'Enabled' if config.get('mcp_config') else 'Disabled'}

=== EXECUTION RESULTS ===
Execution Message: {execution_result.get('message', 'No message available')}
Workflow Received: {execution_result.get('workflow_received', False)}
LLM Config Received: {execution_result.get('llm_config_received', False)}
MCP Config Received: {execution_result.get('mcp_config_received', False)}

=== INPUTS PROVIDED ===
{json.dumps(config.get('inputs', {}), indent=2)}

""".strip()
    
    return workflow_info

def create_task_info(project_id: str, goal: str, additional_info: Dict[str, Any] = None, public_url: str = None) -> dict:
    """Generate comprehensive task info string for a project"""
    task_prompt = TASK_INFO_PROMPT_SUDO.format(
        goal=goal,
        additional_info=additional_info
    )
    
    llm_config = create_llm_config(additional_info.get("llm_config", default_llm_config))
    llm = create_llm_instance(llm_config)
    response = llm.single_generate([{"role": "user", "content": task_prompt}])
    task_info = parse_json_from_text(response)
    task_info = json.loads(task_info[0])
    
    # Add connection_instruction field using the template
    connection_instruction = CONNECTION_INSTRUCTION_PROMPT.format(
        project_id=project_id,
        public_url=public_url or "Not available",
    )
    task_info["connection_instruction"] = connection_instruction
    
    return task_info

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
    
    # Default processing task
    asyncio.create_task(process_stream_task(task_id, config["parameters"]))
    
    return {
        "task_id": task_id,
        "status": "started",
        "stream_url": f"/stream/{task_id}",
        "task_type": task_type
    } 


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

async def execute_workflow_from_config(workflow: Dict[str, Any], llm_config_dict: Dict[str, Any], mcp_config: dict = None, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute a workflow with the given configuration.
    
    Args:
        workflow: The workflow definition/configuration to execute
        llm_config_dict: LLM configuration dictionary
        mcp_config: Optional MCP configuration dictionary
        inputs: Optional inputs dictionary to pass to async_execute
        
    Returns:
        Dict containing execution results and status
        
    """
    try:
        if sudo_execution_result:
            return {
            "status": "completed",
            "message": sudo_execution_result,
            "workflow_received": bool(workflow),
            "llm_config_received": bool(llm_config_dict),
            "mcp_config_received": bool(mcp_config)
        }
        
        
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
        output = await workflow.async_execute(inputs=inputs)
        
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



def generate_project_id() -> str:
    """Generate a unique project ID"""
    return f"proj_{uuid.uuid4().hex[:12]}"

def setup_project(goal: str, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Setup a new project and store it in the project database"""
    project_id = generate_project_id()
    tunnel_info = read_tunnel_info()
    
    # Generate task info with URL information
    task_info = create_task_info(
        project_id, 
        goal, 
        additional_info,
        public_url=tunnel_info.get("public_url") if tunnel_info else None,
    )
    
    # Create project entry
    project_data = {
        "project_id": project_id,
        "goal": goal,
        "additional_info": additional_info or {},
        "created_at": datetime.now().isoformat(),
        "status": "initialized",
        "workflow_generated": False,
        "workflow_executed": False,
        "workflow_data": None,
        "execution_results": None,
        "public_url": tunnel_info.get("public_url") if tunnel_info else None,
        "local_url": tunnel_info.get("local_url") if tunnel_info else None,
        "tunnel_status": tunnel_info.get("status") if tunnel_info else None,
        "task_info": task_info
    }
    
    # Store in project database
    project_info[project_id] = project_data
    
        
    return {
        "project_id": project_id,
        "public_url": project_data["public_url"] or "Not available",
        "local_url": project_data["local_url"] or "Not available", 
        "task_info": task_info
    }

def get_project(project_id: str) -> Dict[str, Any]:
    """Retrieve project information from the database"""
    return project_info.get(project_id)

def update_project_status(project_id: str, status: str, **kwargs):
    """Update project status and other fields"""
    if project_id in project_info:
        project_info[project_id]["status"] = status
        project_info[project_id]["last_updated"] = datetime.now().isoformat()
        
        # Update any additional fields
        for key, value in kwargs.items():
            project_info[project_id][key] = value

def list_projects() -> Dict[str, Any]:
    """List all projects in the database"""
    return {
        "projects": list(project_info.keys()),
        "total_count": len(project_info),
        "active_projects": [
            pid for pid, data in project_info.items() 
            if data.get("status") != "completed"
        ]
    }

async def generate_workflow_for_project(project_id: str, llm_config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate workflow for a specific project.
    Gets goal and workflow specifications from stored project data.
    Uses default_llm_config if no config provided.
    """
    # Check if project exists
    project = get_project(project_id)
    if not project:
        return {
            "success": False,
            "error": "Project not found"
        }
    
    # Get goal from project data
    goal = project.get("goal")
    if not goal:
        return {
            "success": False,
            "error": "Goal not found in project data"
        }
    
    # Get workflow inputs and outputs from task_info
    task_info = project.get("task_info", {})
    print(project)
    workflow_inputs = task_info.get("workflow_inputs", [])
    workflow_outputs = task_info.get("workflow_outputs", [])
    
    # Format the prompt with goal, inputs, and outputs
    formatted_goal = WORKFLOW_GENERATION_PROMPT.format(goal=goal, inputs=workflow_inputs, outputs=workflow_outputs)
    print(formatted_goal)
    
    # Use default config if none provided
    if not llm_config_dict:
        llm_config_dict = default_llm_config
    
    try:
        # Update project status
        update_project_status(project_id, "generating_workflow")
        
        # Use the formatted goal for workflow generation
        workflow_graph = await generate_workflow_from_goal(formatted_goal, llm_config_dict, mcp_config={})
        
        if workflow_graph is None:
            update_project_status(project_id, "generation_failed")
            return {
                "success": False,
                "error": "Failed to generate workflow"
            }
        
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
        
        # Update project with workflow data
        update_project_status(
            project_id, 
            "workflow_generated",
            workflow_generated=True,
            workflow_data=workflow_dict
        )
        
        return {
            "success": True,
            "project_id": project_id,
            "workflow_graph": workflow_dict,
            "workflow_inputs": workflow_inputs,
            "workflow_outputs": workflow_outputs,
            "message": "Workflow generated successfully for project"
        }
        
    except Exception as e:
        update_project_status(project_id, "generation_failed")
        return {
            "success": False,
            "error": f"Error generating workflow: {str(e)}"
        }

async def execute_workflow_for_project(project_id: str, inputs: Dict[str, Any], llm_config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute workflow for a specific project using inputs dictionary.
    Uses default_llm_config if no config provided.
    Gets workflow from project database.
    """
    # Check if project exists
    project = get_project(project_id)
    if not project:
        return {
            "success": False,
            "error": "Project not found"
        }
    
    # Check if workflow was generated for this project
    if not project.get("workflow_generated") or not project.get("workflow_data"):
        return {
            "success": False,
            "error": "No workflow found for this project. Please generate a workflow first."
        }
    
    # Use default config if none provided
    if not llm_config_dict:
        llm_config_dict = default_llm_config
    
    try:
        # Update project status
        update_project_status(project_id, "executing_workflow")
        
        # Get workflow data from project
        workflow_data = project.get("workflow_data")
        
        # Execute the workflow
        execution_result = await execute_workflow_from_config(
            workflow_data, 
            llm_config_dict, 
            mcp_config={}, 
            inputs=inputs
        )
        
        if execution_result is None:
            update_project_status(project_id, "execution_failed")
            return {
                "success": False,
                "error": "Failed to execute workflow"
            }
        
        # Create comprehensive workflow info including public URL
        config_for_workflow_info = {
            "workflow": workflow_data,
            "llm_config": llm_config_dict,
            "mcp_config": {},
            "inputs": inputs
        }
        workflow_info = create_workflow_info(config_for_workflow_info, execution_result)
        
        # Update project with execution results
        update_project_status(
            project_id, 
            "workflow_executed",
            workflow_executed=True,
            execution_results=execution_result,
            workflow_info=workflow_info
        )
        
        return {
            "success": True,
            "project_id": project_id,
            "execution_result": execution_result,
            "workflow_info": workflow_info,
            "inputs": inputs,
            "message": "Workflow executed successfully for project"
        }
        
    except Exception as e:
        update_project_status(project_id, "execution_failed")
        return {
            "success": False,
            "error": f"Error executing workflow: {str(e)}"
        }
