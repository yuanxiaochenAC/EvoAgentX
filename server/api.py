from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse
import json
import asyncio
import uuid
from datetime import datetime

from .models import Config, ProcessResponse, ClientConnectResponse, ClientTaskResponse, ProjectSetupRequest, ProjectSetupResponse, ProjectWorkflowGenerationRequest, ProjectWorkflowExecutionRequest
from .service import handle_process_request, start_streaming_task, setup_project, get_project, list_projects, generate_workflow_for_project, execute_workflow_for_project
from .task_manager import (
    get_stream_task, get_stream_task_updates, is_stream_task_completed,
    create_client_session, get_client_session, get_client_updates, 
    is_client_session_active, add_task_to_client, send_to_client
)

app = FastAPI(title="Processing Server")

@app.post("/process", response_model=ProcessResponse)
async def process_request(config: Config) -> ProcessResponse:
    """
    Process the incoming request with the given configuration.
    Returns a task ID that can be used to retrieve results.
    """
    try:
        return await handle_process_request(config.parameters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New project-based endpoints
@app.post("/project/setup", response_model=ProjectSetupResponse)
async def setup_new_project(request: ProjectSetupRequest) -> ProjectSetupResponse:
    """
    Setup a new project with the given goal and return project information.
    This is the main entry point for creating projects.
    """
    try:
        result = setup_project(request.goal, request.additional_info)
        
        return ProjectSetupResponse(
            project_id=result["project_id"],
            public_url=result["public_url"],
            local_url=result["local_url"],
            task_info=result["task_info"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up project: {str(e)}")


# Project-based workflow generation endpoint
@app.post("/workflow/generate")
async def generate_workflow_for_project_api(request: ProjectWorkflowGenerationRequest):
    """
    Generate a workflow for a specific project.
    Gets inputs and outputs from the stored project data.
    Uses default_llm_config if no config is provided.
    """
    try:
        result = await generate_workflow_for_project(
            request.project_id, 
            request.llm_config
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400 if "not found" in result.get("error", "") else 500,
                detail=result.get("error", "Unknown error")
            )
        
        return {
            "success": True,
            "project_id": result["project_id"],
            "workflow_graph": result["workflow_graph"],
            "workflow_inputs": result["workflow_inputs"],
            "workflow_outputs": result["workflow_outputs"],
            "message": result["message"],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating workflow: {str(e)}")

# Project-based workflow execution endpoint
@app.post("/workflow/execute")
async def execute_workflow_for_project_api(request: ProjectWorkflowExecutionRequest):
    """
    Execute a workflow for a specific project using the inputs dictionary.
    Uses default_llm_config if no config is provided.
    Gets workflow from project database.
    """
    try:
        result = await execute_workflow_for_project(
            request.project_id, 
            request.inputs, 
            request.llm_config
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400 if "not found" in result.get("error", "") else 500,
                detail=result.get("error", "Unknown error")
            )
        
        return {
            "success": True,
            "project_id": result["project_id"],
            "execution_result": result["execution_result"],
            "workflow_info": result["workflow_info"],
            "inputs": result["inputs"],
            "message": result["message"],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing workflow: {str(e)}")

@app.post("/stream/process")
async def start_stream_process(config: Config):
    """
    Start a streaming process and return a task ID.
    """
    return await start_streaming_task(config.dict())





# New client-session endpoints
@app.post("/connect", response_model=ClientConnectResponse)
async def connect_client() -> ClientConnectResponse:
    """
    Create a new client session and return client ID with stream URL.
    """
    client_id = str(uuid.uuid4())
    create_client_session(client_id)
    
    return ClientConnectResponse(
        client_id=client_id,
        stream_url=f"/stream/client/{client_id}"
    )



@app.get("/stream/client/{client_id}")
async def stream_client_updates(client_id: str):
    """
    Persistent SSE stream for a client session.
    """
    if not get_client_session(client_id):
        raise HTTPException(status_code=404, detail="Client session not found")
        
    return EventSourceResponse(
        client_event_generator(client_id, timeout=3600)  # 1 hour timeout
    )

@app.delete("/client/{client_id}")
async def disconnect_client(client_id: str):
    """
    Disconnect a client session.
    """
    if not get_client_session(client_id):
        raise HTTPException(status_code=404, detail="Client session not found")
    
    from .task_manager import close_client_session
    close_client_session(client_id)
    
    return {"status": "disconnected", "client_id": client_id}

async def event_generator(task_id: str, timeout: int = 30):
    """Generate SSE events for a given task"""
    start_time = datetime.now()
    last_index = 0
    
    while True:
        # Check timeout
        if (datetime.now() - start_time).seconds > timeout:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Stream timeout"})
            }
            break
            
        # Check if task exists
        if not get_stream_task(task_id):
            yield {
                "event": "error",
                "data": json.dumps({"error": "Task not found"})
            }
            break
            
        # Get new updates
        updates = get_stream_task_updates(task_id, last_index)
        for update in updates:
            yield {
                "event": "update" if not is_stream_task_completed(task_id) else "complete",
                "data": json.dumps(update)
            }
            last_index += 1
            
        # If task is completed, end stream
        if is_stream_task_completed(task_id):
            break
            
        # Wait before checking for new updates
        await asyncio.sleep(0.5)

async def client_event_generator(client_id: str, timeout: int = 3600):
    """Generate SSE events for a client session"""
    start_time = datetime.now()
    last_index = 0
    
    while True:
        # Check if client session is still active
        if not is_client_session_active(client_id):
            yield {
                "event": "session_closed",
                "data": json.dumps({"message": "Client session closed"})
            }
            break
            
        # Check timeout
        if (datetime.now() - start_time).seconds > timeout:
            yield {
                "event": "session_timeout",
                "data": json.dumps({"message": "Session timed out"})
            }
            break
        
        # Get new updates for this client
        updates = get_client_updates(client_id, last_index)
        for update in updates:
            event_type = update.get("event_type", "update")
            yield {
                "event": event_type,
                "data": json.dumps(update)
            }
            last_index += 1
        
        # Wait before checking for new updates
        await asyncio.sleep(0.5)

@app.get("/stream/{task_id}")
async def stream_results(task_id: str):
    """
    Stream results for a given task ID using Server-Sent Events.
    """
    if not get_stream_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_config = get_stream_task(task_id)["config"]
    return EventSourceResponse(
        event_generator(task_id, timeout=task_config["timeout"])
    )

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

@app.get("/clients")
async def list_clients():
    """List all active client sessions (for debugging)"""
    from .task_manager import client_sessions
    active_clients = []
    
    for client_id, session in client_sessions.items():
        if session["is_active"]:
            active_clients.append({
                "client_id": client_id,
                "created_at": session["created_at"].isoformat(),
                "last_activity": session["last_activity"].isoformat(),
                "active_tasks": session["active_tasks"]
            })
    
    return {"active_clients": active_clients, "total": len(active_clients)} 


@app.get("/project/{project_id}/status")
async def get_project_status(project_id: str):
    """
    Get the current status and information for a specific project.
    """
    project = get_project(project_id)
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return {
        "project_id": project_id,
        "status": project.get("status", "unknown"),
        "goal": project.get("goal", ""),
        "created_at": project.get("created_at", ""),
        "workflow_generated": project.get("workflow_generated", False),
        "workflow_executed": project.get("workflow_executed", False),
        "public_url": project.get("public_url", "Not available"),
        "local_url": project.get("local_url", "Not available"),
        "last_updated": project.get("last_updated", project.get("created_at", ""))
    }

@app.get("/projects")
async def get_all_projects():
    """
    List all projects in the system.
    """
    return list_projects() 