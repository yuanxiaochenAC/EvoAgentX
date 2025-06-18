from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class Config(BaseModel):
    """Configuration model for processing requests"""
    parameters: Dict[str, Any]
    timeout: Optional[int] = 30  # default timeout in seconds

class ProcessResponse(BaseModel):
    """Response model for processing requests"""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

# New models for client session functionality
class ClientConnectResponse(BaseModel):
    """Response model for client connection"""
    client_id: str
    stream_url: str

class ClientTaskResponse(BaseModel):
    """Response model for client task start"""
    task_id: str
    status: str
    client_id: str

# New models for project-based approach
class ProjectSetupRequest(BaseModel):
    """Request model for project setup"""
    goal: str
    additional_info: Optional[Dict[str, Any]] = None  # Any additional project information

class ProjectSetupResponse(BaseModel):
    """Response model for project setup"""
    project_id: str
    public_url: str
    task_info: str  # Contains connection_instruction from the LLM-generated task info

class ProjectWorkflowGenerationRequest(BaseModel):
    """Request model for project-based workflow generation"""
    project_id: str
    llm_config: Optional[Dict[str, Any]] = None  # Optional, will use default if not provided

class ProjectWorkflowExecutionRequest(BaseModel):
    """Request model for project-based workflow execution"""
    project_id: str
    inputs: Dict[str, Any]  # Inputs dictionary to pass to workflow execution
    llm_config: Optional[Dict[str, Any]] = None  # Optional, will use default if not provided 