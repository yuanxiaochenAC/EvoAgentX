"""
API routes for EvoAgentX application.
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from typing import List, Optional, Dict, Any

from datetime import timedelta

from evoagentx.app.config import settings
from evoagentx.app.schemas import (
    AgentCreate, AgentUpdate, AgentResponse,
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    ExecutionCreate, ExecutionResponse,
    PaginationParams, SearchParams,
    Token, UserCreate, UserLogin, UserResponse
)
from evoagentx.app.services import AgentService, WorkflowService
from evoagentx.app.security import (
    create_access_token, 
    authenticate_user, 
    create_user, 
    get_current_active_user,
    get_current_admin_user
)
from evoagentx.app.db import Database

# Create routers for different route groups
auth_router = APIRouter(prefix=settings.API_PREFIX)
agents_router = APIRouter(prefix=settings.API_PREFIX)
workflows_router = APIRouter(prefix=settings.API_PREFIX)
executions_router = APIRouter(prefix=settings.API_PREFIX)
system_router = APIRouter(prefix=settings.API_PREFIX)

# Authentication Routes
@auth_router.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(user: UserCreate):
    """Register a new user."""
    return await create_user(user)

@auth_router.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and return access token."""
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user['email'], 
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer"
    }

# Agent Routes
@agents_router.post("/agents", response_model=AgentResponse, tags=["Agents"])
async def create_agent(
    agent: AgentCreate, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Create a new agent."""
    try:
        created_agent = await AgentService.create_agent(
            agent, 
            user_id=str(current_user['_id'])
        )
        return AgentResponse(**created_agent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@agents_router.get("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def get_agent(
    agent_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Retrieve a specific agent by ID."""
    agent = await AgentService.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentResponse(**agent)

@agents_router.put("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def update_agent(
    agent_id: str, 
    agent_update: AgentUpdate, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Update an existing agent."""
    try:
        updated_agent = await AgentService.update_agent(agent_id, agent_update)
        if not updated_agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return AgentResponse(**updated_agent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@agents_router.delete("/agents/{agent_id}", tags=["Agents"])
async def delete_agent(
    agent_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_admin_user)
):
    """Delete an agent (admin-only)."""
    try:
        success = await AgentService.delete_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"message": "Agent deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@agents_router.get("/agents", response_model=List[AgentResponse], tags=["Agents"])
async def list_agents(
    pagination: PaginationParams = Depends(),
    search: Optional[SearchParams] = Depends(),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """List agents with optional pagination and search."""
    agents, total = await AgentService.list_agents(pagination, search)
    
    # You might want to add total count to response headers or as a separate field
    return [AgentResponse(**agent) for agent in agents]

# Workflow Routes
@workflows_router.post("/workflows", response_model=WorkflowResponse, tags=["Workflows"])
async def create_workflow(
    workflow: WorkflowCreate, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Create a new workflow."""
    try:
        created_workflow = await WorkflowService.create_workflow(
            workflow, 
            user_id=str(current_user['_id'])
        )
        return WorkflowResponse(**created_workflow)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@workflows_router.get("/workflows/{workflow_id}", response_model=WorkflowResponse, tags=["Workflows"])
async def get_workflow(
    workflow_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Retrieve a specific workflow by ID."""
    workflow = await WorkflowService.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return WorkflowResponse(**workflow)

@workflows_router.put("/workflows/{workflow_id}", response_model=WorkflowResponse, tags=["Workflows"])
async def update_workflow(
    workflow_id: str, 
    workflow_update: WorkflowUpdate, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Update an existing workflow."""
    try:
        updated_workflow = await WorkflowService.update_workflow(workflow_id, workflow_update)
        if not updated_workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return WorkflowResponse(**updated_workflow)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@workflows_router.delete("/workflows/{workflow_id}", tags=["Workflows"])
async def delete_workflow(
    workflow_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_admin_user)
):
    """Delete a workflow (admin-only)."""
    try:
        success = await WorkflowService.delete_workflow(workflow_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {"message": "Workflow deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@workflows_router.get("/workflows", response_model=List[WorkflowResponse], tags=["Workflows"])
async def list_workflows(
    pagination: PaginationParams = Depends(),
    search: Optional[SearchParams] = Depends(),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """List workflows with optional pagination and search."""
    workflows, total = await WorkflowService.list_workflows(pagination, search)
    
    return [WorkflowResponse(**workflow) for workflow in workflows]

# Workflow Execution Routes
@executions_router.post("/executions", response_model=ExecutionResponse, tags=["Executions"])
async def create_execution(
    execution: ExecutionCreate, 
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Create and start a workflow execution."""
    try:
        execution_result = await WorkflowService.create_execution(
            execution, 
            user_id=str(current_user['_id']),
            background_tasks=background_tasks
        )
        return ExecutionResponse(**execution_result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@executions_router.get("/executions/{execution_id}", response_model=ExecutionResponse, tags=["Executions"])
async def get_execution(
    execution_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Retrieve a specific workflow execution by ID."""
    execution = await WorkflowService.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return ExecutionResponse(**execution)

@executions_router.get("/executions", response_model=List[ExecutionResponse], tags=["Executions"])
async def list_executions(
    pagination: PaginationParams = Depends(),
    search: Optional[SearchParams] = Depends(),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """List workflow executions with optional pagination and search."""
    executions, total = await WorkflowService.list_executions(pagination, search)
    
    return [ExecutionResponse(**execution) for execution in executions]

# Health Check Route
@system_router.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    try:
        # You can add more comprehensive health checks here
        await Database.db.command('ping')
        return {
            "status": "healthy", 
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# Export the routers
__all__ = [
    'auth_router',
    'agents_router',
    'workflows_router',
    'executions_router',
    'system_router'
]