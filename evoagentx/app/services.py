"""
Business logic for agents, workflows, and executions.
"""
import logging
import os
import json
import uuid
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from bson import ObjectId

from evoagentx.models.model_configs import OpenAILLMConfig, LLMConfig
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.models.litellm_model import LiteLLM
from evoagentx.models.siliconflow_model import SiliconFlowLLM

from evoagentx.app.db import (
    database, AgentStatus, WorkflowStatus, ExecutionStatus
)
from evoagentx.app.schemas import (
    AgentCreate, AgentUpdate, WorkflowCreate, WorkflowUpdate, 
    ExecutionCreate, PaginationParams, SearchParams, AgentQueryRequest
)
from evoagentx.app.backbone import goal_based_workflow_generation, workflow_execution

logger = logging.getLogger(__name__)

# Agent Service
class AgentService:
    @staticmethod
    async def create_agent(agent_data: AgentCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new agent."""
        agent_dict = agent_data.dict()
        agent_dict["created_by"] = user_id
        agent_dict["created_at"] = datetime.utcnow()
        agent_dict["updated_at"] = agent_dict["created_at"]
        agent_dict["status"] = AgentStatus.CREATED
        
        # Validate agent exists with the same name
        existing_agent = await database.find_one("agents", {"name": agent_dict["name"]})
        if existing_agent:
            raise ValueError(f"Agent with name '{agent_dict['name']}' already exists")
        
        try:
            result_id = await database.write("agents", agent_dict)
            agent_dict["_id"] = result_id
            logger.info(f"Created agent {agent_dict['name']} with ID {result_id}")
            return agent_dict
        except Exception as e:
            logger.error(f"Failed to add agent {agent_dict['name']} to Database: {e}, agent_dict: {agent_dict}")
            raise ValueError(f"Failed to add agent to Database: {e}")
    
    @staticmethod
    async def get_agent(agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        agent = await database.find_one("agents", {"_id": ObjectId(agent_id)})
        return agent
    
    @staticmethod
    async def get_agent_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get an agent by name."""
        return await database.find_one("agents", {"name": name})
    
    @staticmethod
    async def update_agent(agent_id: str, agent_data: AgentUpdate) -> Optional[Dict[str, Any]]:
        """Update an agent."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        agent = await database.find_one("agents", {"_id": ObjectId(agent_id)})
        if not agent:
            return None
        
        update_data = agent_data.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        if "name" in update_data:
            # Check if the new name already exists
            existing = await database.find_one("agents", {
                "name": update_data["name"],
                "_id": {"$ne": ObjectId(agent_id)}
            })
            if existing:
                raise ValueError(f"Agent with name '{update_data['name']}' already exists")
        
        # Handle partial config updates
        if "config" in update_data and isinstance(update_data["config"], dict) and isinstance(agent.get("config"), dict):
            # Instead of replacing the entire config, merge the updated fields with existing ones
            merged_config = agent["config"].copy()
            for key, value in update_data["config"].items():
                merged_config[key] = value
            update_data["config"] = merged_config
            logger.info(f"Merged config update for agent {agent_id}: {update_data['config']}")
        
        # Update in database
        await database.update(
            "agents",
            {"_id": ObjectId(agent_id)},
            {"$set": update_data}
        )
        
        # Get the updated agent with all fields merged
        updated_agent = await database.find_one("agents", {"_id": ObjectId(agent_id)})
        logger.info(f"Updated agent {agent_id}")
        
        return updated_agent
    
    @staticmethod
    async def delete_agent(agent_id: str) -> bool:
        """Delete an agent."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
          
        # Check if agent is used in any workflows
        workflow_count = await database.count("workflows", {"agent_ids": agent_id})
        if workflow_count > 0:
            raise ValueError(f"Cannot delete agent {agent_id} as it is used in {workflow_count} workflows")

        result = await database.delete("agents", {"_id": ObjectId(agent_id)})
        if result > 0:
            logger.info(f"Deleted agent {agent_id}")
            return True
        return False
    
    @staticmethod
    async def list_agents(
        params: PaginationParams, 
        search: Optional[SearchParams] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List agents with pagination and search."""
        query = {}
        
        if search:
            if search.query:
                query["$text"] = {"$search": search.query}
            
            if search.tags:
                query["tags"] = {"$all": search.tags}
            
            if search.status:
                query["status"] = search.status
            
            if search.start_date and search.end_date:
                query["created_at"] = {
                    "$gte": search.start_date,
                    "$lte": search.end_date
                }
            elif search.start_date:
                query["created_at"] = {"$gte": search.start_date}
            elif search.end_date:
                query["created_at"] = {"$lte": search.end_date}
        
        total = await database.count("agents", query)
        
        agents = await database.search("agents", query, 
                                skip=params.skip, 
                                limit=params.limit, 
                                sort=[("created_at", -1)])
        
        return agents, total
    
    @staticmethod
    async def query_agent(agent_id: str, query: AgentQueryRequest, user_id: Optional[str] = None) -> str:
        """Query an agent with a prompt and get a response."""
        # Get the agent from database
        agent_data = await database.find_one("agents", {"_id": ObjectId(agent_id)})
        if not agent_data:
            raise ValueError(f"Agent with ID {agent_id} not found")

        try:
            # Get the model type from config
            llm_type = agent_data["config"]["llm_type"].lower()
            
            # Create LLM config
            llm_config = LLMConfig(
                llm_type=agent_data["config"]["llm_type"],
                model=agent_data["config"]["model"],
                openai_key=agent_data["config"]["openai_key"],
                temperature=agent_data["config"]["temperature"],
                max_tokens=agent_data["config"]["max_tokens"],
                top_p=agent_data["config"]["top_p"],
                output_response=agent_data["config"]["output_response"]
            )
            
            # Initialize the appropriate LLM based on type
            if llm_type == "openaillm":
                llm = OpenAILLM(config=llm_config)
            elif llm_type == "litellm":
                llm = LiteLLM(config=llm_config)
            elif llm_type == "siliconflow":
                llm = SiliconFlowLLM(config=llm_config)
            else:
                raise ValueError(f"Unsupported model type: {llm_type}")
            
            # Initialize the model
            llm.init_model()
            
            # Format messages with history
            messages = []
            
            # Add system message if present
            system_prompt = agent_data["config"].get("prompt", "")
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history if provided
            if query.history:
                messages.extend(query.history)
            # Add the current user message
            messages.append({"role": "user", "content": query.prompt})
  
            # Generate response using single_generate
            response = llm.single_generate(messages=messages)
            return response
            
        except Exception as e:
            logger.error(f"Error querying agent {agent_id}: {str(e)}")
            raise ValueError(f"Error querying agent: {str(e)}")

# Workflow Service
class WorkflowService:
    @staticmethod
    async def create_workflow(workflow_data: WorkflowCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new workflow."""
        workflow_dict = workflow_data.dict()
        workflow_dict["created_by"] = user_id
        workflow_dict["created_at"] = datetime.utcnow()
        workflow_dict["updated_at"] = workflow_dict["created_at"]
        workflow_dict["status"] = WorkflowStatus.CREATED
        workflow_dict["version"] = 1
        
        # Extract agent IDs from the workflow definition
        agent_ids = set()
        
        # Extract agent IDs from steps
        steps = workflow_dict["definition"].get("steps", [])
        for step in steps:
            if "agent_id" in step:
                agent_id = step["agent_id"]
                # Validate agent exists
                agent = await AgentService.get_agent(agent_id)
                if not agent:
                    raise ValueError(f"Agent with ID {agent_id} does not exist")
                agent_ids.add(agent_id)
        
        workflow_dict["agent_ids"] = list(agent_ids)
        
        # Check for existing workflow with the same name
        existing = await database.find_one("workflows", {"name": workflow_dict["name"]})
        if existing:
            raise ValueError(f"Workflow with name '{workflow_dict['name']}' already exists")
        
        result_id = await database.write("workflows", workflow_dict)
        workflow_dict["_id"] = result_id
        
        logger.info(f"Created workflow {workflow_dict['name']} with ID {result_id}")
        
        return workflow_dict
    
    @staticmethod
    async def get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by ID."""
        if not ObjectId.is_valid(workflow_id):
            raise ValueError(f"Invalid workflow ID: {workflow_id}")
        workflow = await database.find_one("workflows", {"_id": ObjectId(workflow_id)})
        return workflow
    
    @staticmethod
    async def get_workflow_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by name."""
        return await database.find_one("workflows", {"name": name})
    
    @staticmethod
    async def update_workflow(workflow_id: str, workflow_data: WorkflowUpdate) -> Optional[Dict[str, Any]]:
        """Update a workflow."""
        if not ObjectId.is_valid(workflow_id):
            raise ValueError(f"Invalid workflow ID: {workflow_id}")
            
        workflow = await database.find_one("workflows", {"_id": ObjectId(workflow_id)})
        if not workflow:
            return None
        
        update_data = workflow_data.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Update version if definition changes
        if "definition" in update_data:
            update_data["version"] = workflow.get("version", 1) + 1
            
            # Extract agent IDs from the updated workflow definition
            agent_ids = set()
            steps = update_data["definition"].get("steps", [])
            for step in steps:
                if "agent_id" in step:
                    agent_id = step["agent_id"]
                    # Validate agent exists
                    agent = await AgentService.get_agent(agent_id)
                    if not agent:
                        raise ValueError(f"Agent with ID {agent_id} does not exist")
                    agent_ids.add(agent_id)
            
            update_data["agent_ids"] = list(agent_ids)
        
        # Check for name conflict if name is being updated
        if "name" in update_data:
            existing = await database.find_one("workflows", {
                "name": update_data["name"],
                "_id": {"$ne": ObjectId(workflow_id)}
            })
            if existing:
                raise ValueError(f"Workflow with name '{update_data['name']}' already exists")
        
        await database.update(
            "workflows",
            {"_id": ObjectId(workflow_id)},
            {"$set": update_data}
        )
        
        updated_workflow = await database.find_one("workflows", {"_id": ObjectId(workflow_id)})
        logger.info(f"Updated workflow {workflow_id}")
        
        return updated_workflow
    
    @staticmethod
    async def delete_workflow(workflow_id: str) -> bool:
        """Delete a workflow."""
        if not ObjectId.is_valid(workflow_id):
            raise ValueError(f"Invalid workflow ID: {workflow_id}")
        
        # Find all executions for the workflow
        executions = await database.search("executions", {"workflow_id": workflow_id})
        
        # Separate pending and running executions
        pending_executions = [exec for exec in executions if exec["status"] == ExecutionStatus.PENDING]
        running_executions = [exec for exec in executions if exec["status"] == ExecutionStatus.RUNNING]
        
        # If there are running executions, raise an error
        if running_executions:
            raise ValueError(f"Cannot delete workflow {workflow_id} with {len(running_executions)} running executions")
        
        # Stop and remove pending executions
        for exec in pending_executions:
            await WorkflowExecutionService.update_execution_status(
                execution_id=str(exec["_id"]),
                status=ExecutionStatus.CANCELLED
            )
            await database.delete("executions", {"_id": exec["_id"]})
            logger.info(f"Stopped and removed pending execution {exec['_id']} for workflow {workflow_id}")
        
        # Delete the workflow
        result = await database.delete("workflows", {"_id": ObjectId(workflow_id)})
        if result > 0:
            # Optionally, delete associated logs
            await database.delete("logs", {"workflow_id": workflow_id})
            logger.info(f"Deleted workflow {workflow_id}")
            return True
        
        return False
    
    @staticmethod
    async def list_workflows(
        params: PaginationParams, 
        search: Optional[SearchParams] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List workflows with pagination and search."""
        query = {}
        
        if search:
            if search.query:
                query["$text"] = {"$search": search.query}
            
            if search.tags:
                query["tags"] = {"$all": search.tags}
            
            if search.status:
                query["status"] = search.status
            
            if search.start_date and search.end_date:
                query["created_at"] = {
                    "$gte": search.start_date,
                    "$lte": search.end_date
                }
            elif search.start_date:
                query["created_at"] = {"$gte": search.start_date}
            elif search.end_date:
                query["created_at"] = {"$lte": search.end_date}
        
        total = await database.count("workflows", query)
        
        workflows = await database.search("workflows", query, 
                                         skip=params.skip, 
                                         limit=params.limit, 
                                         sort=[("created_at", -1)])
        return workflows, total

# Workflow Execution Service
class WorkflowExecutionService:
    @staticmethod
    async def create_execution(execution_data: ExecutionCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new workflow execution."""
        # Validate workflow exists
        workflow = await WorkflowService.get_workflow(execution_data.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {execution_data.workflow_id} not found")
        
        # Prepare execution document
        execution_dict = {
            "workflow_id": execution_data.workflow_id,
            "status": ExecutionStatus.PENDING,
            "start_time": datetime.utcnow(),
            "input_params": execution_data.input_params,
            "created_by": user_id,
            "created_at": datetime.utcnow(),
            "step_results": {},
            "current_step": None,
            "results": {},
            "error_message": None
        }
        
        # Insert execution record
        result_id = await database.write("executions", execution_dict)
        execution_dict["_id"] = result_id
        
        logger.info(f"Created workflow execution {result_id}")
        
        return execution_dict
    
    @staticmethod
    async def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow execution by ID."""
        if not ObjectId.is_valid(execution_id):
            raise ValueError(f"Invalid execution ID: {execution_id}")
            
        execution = await database.find_one("executions", {"_id": ObjectId(execution_id)})
        return execution
    
    @staticmethod
    async def update_execution_status(execution_id: str, status: ExecutionStatus, error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Update execution status."""
        if not ObjectId.is_valid(execution_id):
            raise ValueError(f"Invalid execution ID: {execution_id}")
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
            update_data["end_time"] = datetime.utcnow()
        
        if error_message:
            update_data["error_message"] = error_message
        
        result = await database.find_one_and_update(
            "executions",
            {"_id": ObjectId(execution_id)},
            {"$set": update_data}
        )
        
        return result
    
    @staticmethod
    async def update_execution(execution_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update execution with arbitrary data."""
        if not ObjectId.is_valid(execution_id):
            raise ValueError(f"Invalid execution ID: {execution_id}")
        
        # Always update the timestamp
        update_data["updated_at"] = datetime.utcnow()
        
        result = await database.find_one_and_update(
            "executions",
            {"_id": ObjectId(execution_id)},
            {"$set": update_data}
        )
        
        return result
    
    @staticmethod
    async def list_executions(
        workflow_id: Optional[str] = None,
        params: PaginationParams = PaginationParams(), 
        search: Optional[SearchParams] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List workflow executions with pagination and search."""
        query = {}
        
        if workflow_id:
            query["workflow_id"] = workflow_id
        
        if search:
            if search.status:
                query["status"] = search.status
            
            if search.start_date and search.end_date:
                query["created_at"] = {
                    "$gte": search.start_date,
                    "$lte": search.end_date
                }
            elif search.start_date:
                query["created_at"] = {"$gte": search.start_date}
            elif search.end_date:
                query["created_at"] = {"$lte": search.end_date}
        
        total = await database.count("executions", query)
        
        executions = await database.search("executions", query, 
                                          skip=params.skip, 
                                          limit=params.limit, 
                                          sort=[("created_at", -1)])
        return executions, total
    
    @staticmethod
    async def log_execution_event(
        workflow_id: str, 
        execution_id: str, 
        message: str,
        step_id: Optional[str] = None, 
        agent_id: Optional[str] = None, 
        level: str = "INFO", 
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Log an event in a workflow execution."""
        log_entry = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "step_id": step_id,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow(),
            "level": level,
            "message": message,
            "details": details or {}
        }
        
        result_id = await database.write("logs", log_entry)
        log_entry["_id"] = result_id
        
        return log_entry
    
    @staticmethod
    async def get_execution_logs(
        execution_id: str, 
        params: PaginationParams = PaginationParams()
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Retrieve logs for a specific execution."""
        query = {"execution_id": execution_id}
        
        total = await database.count("logs", query)
        
        logs = await database.search("logs", query, 
                                    skip=params.skip, 
                                    limit=params.limit, 
                                    sort=[("timestamp", 1)])
        return logs, total

class AgentBackupService:
    @staticmethod
    async def save_agent_backup(agent_id: str, backup_path: str) -> Dict[str, Any]:
        """Save an agent's current state to a backup file."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        # Get the agent from database
        agent = await database.find_one("agents", {"_id": ObjectId(agent_id)})
        if not agent:
            raise ValueError(f"Agent with ID {agent_id} not found in database")
        
        # Create backup directory if it doesn't exist
        backup_dir = os.path.dirname(backup_path)
        if backup_dir and not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Save the agent to the specified path
        try:
            with open(backup_path, 'w') as f:
                # Convert ObjectId to string for JSON serialization
                agent_copy = dict(agent)
                agent_copy["_id"] = str(agent_copy["_id"])
                json.dump(agent_copy, f, indent=2, default=str)
            
            return {
                "success": True,
                "message": f"Agent backup saved to {backup_path}",
                "agent_id": agent_id,
                "backup_path": backup_path,
                "agent_name": agent["name"]
            }
        except Exception as e:
            logger.error(f"Failed to save agent backup: {str(e)}")
            raise ValueError(f"Failed to save agent backup: {str(e)}")
    
    @staticmethod
    async def restore_agent_backup(backup_path: str) -> Dict[str, Any]:
        """Restore an agent from a backup file and create it in the database."""
        if not os.path.exists(backup_path):
            raise ValueError(f"Backup file not found: {backup_path}")
            
        try:
            # Load the backup file directly as JSON
            with open(backup_path, 'r') as f:
                agent_data = json.load(f)
                
            # Basic validation
            required_fields = ["name", "description"]
            for field in required_fields:
                if field not in agent_data:
                    raise ValueError(f"Missing required field '{field}' in backup file")
            
            # Create a new agent with the data from the backup
            original_name = agent_data["name"]
            
            # Remove any existing numeric suffix
            if '_' in original_name:
                base_name = original_name.split('_')[0]
            else:
                base_name = original_name
                
            # Generate a new UUID suffix - use just the last 8 characters
            uuid_suffix = uuid.uuid4().hex[-8:]
            new_name = f"{base_name}_{uuid_suffix}"
            
            # Remove _id field if it exists (we'll generate a new one)
            if "_id" in agent_data:
                del agent_data["_id"]
            
            # Update the name
            agent_data["name"] = new_name
            agent_data["created_at"] = datetime.utcnow()
            agent_data["updated_at"] = agent_data["created_at"]
            agent_data["status"] = AgentStatus.CREATED
            
            # Create the agent using AgentService
            result_id = await database.write("agents", agent_data)
            agent_data["_id"] = result_id
            
            return {
                "success": True,
                "message": f"Agent restored from {backup_path} with new name {new_name}",
                "agent_id": str(result_id),
                "backup_path": backup_path,
                "agent_name": new_name
            }
        except Exception as e:
            logger.error(f"Failed to restore agent from backup: {str(e)}")
            raise ValueError(f"Failed to restore agent from backup: {str(e)}")
    
    @staticmethod
    async def list_agent_backups(agent_id: str, backup_dir: str) -> List[Dict[str, Any]]:
        """List all backup files for an agent."""
        if not ObjectId.is_valid(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
            
        # Get the agent from database
        agent = await database.find_one("agents", {"_id": ObjectId(agent_id)})
        if not agent:
            raise ValueError(f"Agent with ID {agent_id} not found")
            
        try:
            # Create backup directory if it doesn't exist
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            # Get all backup files for this agent
            backup_pattern = os.path.join(backup_dir, f"{agent['name']}_*.json")
            backup_files = glob.glob(backup_pattern)
            
            # Get file details
            backups = []
            for file_path in backup_files:
                file_stat = os.stat(file_path)
                backups.append({
                    "path": file_path,
                    "created_at": datetime.fromtimestamp(file_stat.st_ctime),
                    "size": file_stat.st_size
                })
            
            # Sort by creation date, newest first
            backups.sort(key=lambda x: x["created_at"], reverse=True)
            
            return backups
        except Exception as e:
            logger.error(f"Failed to list agent backups: {str(e)}")
            raise ValueError(f"Failed to list agent backups: {str(e)}")
    
    @staticmethod
    async def backup_all_agents(backup_dir: str) -> Dict[str, Any]:
        """
        Backup all agents in the database to the specified directory.
        
        Args:
            backup_dir: Directory where agent backups will be stored
            
        Returns:
            Dict with results of the batch backup operation
        """
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        # Get all agents from the database
        agents = await database.search("agents", {})
        
        results = []
        success_count = 0
        error_count = 0
        
        # Backup each agent
        for agent in agents:
            try:
                agent_id = str(agent["_id"])
                agent_name = agent["name"]
                backup_path = os.path.join(backup_dir, f"{agent_name}_backup.json")
                
                result = await AgentBackupService.save_agent_backup(agent_id, backup_path)
                results.append(result)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to back up agent {agent.get('name', 'unknown')}: {str(e)}")
                results.append({
                    "success": False,
                    "agent_name": agent.get("name", "unknown"),
                    "error": str(e)
                })
                error_count += 1
                
        return {
            "success": error_count == 0,
            "total": len(agents),
            "successful": success_count,
            "failed": error_count,
            "backup_dir": backup_dir,
            "results": results
        }
        
    @staticmethod
    async def backup_agents(agent_ids: List[str], backup_dir: str) -> Dict[str, Any]:
        """
        Backup specific agents to the specified directory.
        
        Args:
            agent_ids: List of agent IDs to back up
            backup_dir: Directory where agent backups will be stored
            
        Returns:
            Dict with results of the batch backup operation
        """
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        results = []
        success_count = 0
        error_count = 0
        
        # Backup each specified agent
        for agent_id in agent_ids:
            try:
                # Get agent from DB to get its name
                agent = await database.find_one("agents", {"_id": ObjectId(agent_id)})
                if not agent:
                    results.append({
                        "success": False,
                        "agent_id": agent_id,
                        "error": f"Agent with ID {agent_id} not found in database"
                    })
                    error_count += 1
                    continue
                    
                agent_name = agent["name"]
                backup_path = os.path.join(backup_dir, f"{agent_name}_backup.json")
                
                result = await AgentBackupService.save_agent_backup(agent_id, backup_path)
                results.append(result)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to back up agent {agent_id}: {str(e)}")
                results.append({
                    "success": False,
                    "agent_id": agent_id,
                    "error": str(e)
                })
                error_count += 1
                
        return {
            "success": error_count == 0,
            "total": len(agent_ids),
            "successful": success_count,
            "failed": error_count,
            "backup_dir": backup_dir,
            "results": results
        }
        
    @staticmethod
    async def restore_agents_from_files(backup_files: List[str]) -> Dict[str, Any]:
        """
        Restore multiple agents from a list of backup files.
        
        Args:
            backup_files: List of paths to agent backup files
            
        Returns:
            Dict with results of the batch restore operation
        """
        results = []
        success_count = 0
        error_count = 0
        
        # Restore each agent from the provided backup files
        for backup_path in backup_files:
            try:
                result = await AgentBackupService.restore_agent_backup(backup_path)
                results.append(result)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to restore agent from {backup_path}: {str(e)}")
                results.append({
                    "success": False,
                    "backup_path": backup_path,
                    "error": str(e)
                })
                error_count += 1
                
        return {
            "success": error_count == 0,
            "total": len(backup_files),
            "successful": success_count,
            "failed": error_count,
            "results": results
        }
        
    @staticmethod
    async def restore_all_agents_from_directory(backup_dir: str) -> Dict[str, Any]:
        """
        Restore all agents from backup files in a directory.
        
        Args:
            backup_dir: Directory containing agent backup files
            
        Returns:
            Dict with results of the batch restore operation
        """
        if not os.path.exists(backup_dir):
            raise ValueError(f"Backup directory not found: {backup_dir}")
            
        # Find all JSON files in the directory
        backup_files = glob.glob(os.path.join(backup_dir, "*.json"))
        
        if not backup_files:
            return {
                "success": True,
                "total": 0,
                "message": f"No backup files found in {backup_dir}"
            }
            
        # Restore all agents from the found backup files
        return await AgentBackupService.restore_agents_from_files(backup_files)

class WorkflowGeneratorService:
    @staticmethod
    async def generate_workflow(goal: str, llm_config: Dict[str, Any], additional_info: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a workflow graph based on a goal and LLM configuration using the backbone system.
        Also stores the generated workflow in the database.
        
        Args:
            goal: The goal or task description for the workflow
            llm_config: Configuration for the language model
            additional_info: Optional additional context or instructions
            user_id: ID of the user creating the workflow
            
        Returns:
            A result dict containing the workflow graph, task information, and database ID
        """
        try:
            # Use the backbone goal-based workflow generation
            logger.info(f"Generating workflow using backbone for goal: {goal}")
            result = await goal_based_workflow_generation(
                goal=goal,
                llm_config=llm_config,
                additional_info=additional_info
            )
            
            # If generation was successful and we have a workflow graph, store it in database
            if result.get("success") and result.get("workflow_graph"):
                try:
                    # Extract workflow information - workflow_graph is always a dict from backbone
                    workflow_graph_dict = result["workflow_graph"]
                    task_info = result.get("task_info", {})
        
                    # Create workflow name and description
                    workflow_name = task_info.get("workflow_name", f"Generated Workflow for: {goal[:50]}...")
                    workflow_description = task_info.get("workflow_description", f"Auto-generated workflow from goal: {goal}")
        
                    # The workflow_graph is already a dictionary from backbone, ready for storage
                    workflow_definition = workflow_graph_dict
                    
                    # Create WorkflowCreate object and store using WorkflowService
                    workflow_create = WorkflowCreate(
                        name=workflow_name,
                        description=workflow_description,
                        definition=workflow_definition,
                        tags=["auto-generated", "goal-based"]
                    )
                    
                    created_workflow = await WorkflowService.create_workflow(
                        workflow_data=workflow_create,
                        user_id=user_id
                    )
                    
                    workflow_id = str(created_workflow["_id"])
                    
                    result["workflow_id"] = workflow_id
                    result["created_at"] = datetime.utcnow()
                    result["created_by"] = user_id
                    logger.info(f"Successfully stored generated workflow with ID: {workflow_id}")
                except Exception as e:
                    logger.warning(f"Failed to store workflow in database: {str(e)}")
                    # Don't fail the entire generation if storage fails
                    result["storage_warning"] = f"Workflow generated but not stored: {str(e)}"
            
            return result
        except Exception as e:
            logger.error(f"Failed to generate workflow: {str(e)}")
            raise ValueError(f"Failed to generate workflow: {str(e)}")
    

    @staticmethod
    async def execute_workflow(workflow_graph, llm_config: Dict[str, Any], inputs: Dict[str, Any], mcp_config: Optional[Dict[str, Any]] = None, workflow_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a workflow graph with the given configuration.
        Also stores execution records in the database.
        
        Args:
            workflow_graph: The workflow graph object (WorkFlowGraph) or dict to execute
            llm_config: LLM configuration dictionary
            inputs: Input parameters for the workflow execution
            mcp_config: Optional MCP configuration dictionary
            workflow_id: Optional workflow ID if executing a stored workflow
            user_id: ID of the user executing the workflow
            
        Returns:
            Dict containing execution results, status, and database IDs
        """
        execution_id = None
        try:
            # Create execution record first using WorkflowExecutionService
            if workflow_id:
                try:
                    execution_create = ExecutionCreate(
                        workflow_id=workflow_id,
                        input_params=inputs
                    )
                    
                    execution_record = await WorkflowExecutionService.create_execution(
                        execution_data=execution_create,
                        user_id=user_id
                    )
                    execution_id = str(execution_record["_id"])
                    
                    # Update status to running
                    await WorkflowExecutionService.update_execution_status(
                        execution_id=execution_id,
                        status=ExecutionStatus.RUNNING
                    )
                    logger.info(f"Created execution record with ID: {execution_id}")
                except Exception as e:
                    logger.warning(f"Failed to create execution record: {e}")
            
            # Use the backbone workflow execution
            logger.info("Executing workflow using backbone")
            result = await workflow_execution(
                workflow_graph=workflow_graph,
                llm_config=llm_config,
                inputs=inputs,
                mcp_config=mcp_config
            )
            
            # Update execution record with results using WorkflowExecutionService
            if execution_id and result.get("success"):
                try:
                    # Update status and results in one call
                    await WorkflowExecutionService.update_execution(
                        execution_id=execution_id,
                        update_data={
                            "status": ExecutionStatus.COMPLETED,
                            "end_time": datetime.utcnow(),
                            "results": result
                        }
                    )
                    logger.info(f"Updated execution record {execution_id} with success")
                except Exception as e:
                    logger.warning(f"Failed to update execution record: {e}")
            elif execution_id and not result.get("success"):
                try:
                    await WorkflowExecutionService.update_execution_status(
                        execution_id=execution_id,
                        status=ExecutionStatus.FAILED,
                        error_message=result.get("error", "Unknown error")
                    )
                    logger.info(f"Updated execution record {execution_id} with failure")
                except Exception as e:
                    logger.warning(f"Failed to update execution record: {e}")
            
            # Add database IDs to result
            if workflow_id:
                result["workflow_id"] = workflow_id
            if execution_id:
                result["execution_id"] = execution_id
                result["created_at"] = datetime.utcnow()
            
            return result
        except Exception as e:
            # Update execution record with error if it exists
            if execution_id:
                try:
                    await WorkflowExecutionService.update_execution(
                        execution_id=execution_id,
                        update_data={
                            "status": ExecutionStatus.FAILED,
                            "end_time": datetime.utcnow(),
                            "error_message": str(e)
                        }
                    )
                except Exception as update_error:
                    logger.warning(f"Failed to update execution record with error: {update_error}")
            
            logger.error(f"Failed to execute workflow: {str(e)}")
            raise ValueError(f"Failed to execute workflow: {str(e)}")