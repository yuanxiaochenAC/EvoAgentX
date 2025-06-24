import json
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from evoagentx.models.model_configs import LLMConfig, OpenAILLMConfig
from evoagentx.models.model_utils import create_llm_instance
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.core.module_utils import parse_json_from_text
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.tools import MCPToolkit
from .prompts import TASK_INFO_PROMPT, WORKFLOW_GENERATION_PROMPT

logger = logging.getLogger(__name__)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
default_llm_config = {
    "model": "gpt-4o-mini",
    "openai_key": OPENAI_API_KEY,
    "stream": True,
    "output_response": True,
    "max_tokens": 16000
}
# default_llm_config = None
# sudo_workflow = WorkFlow.from_file("examples/output/jobs/jobs_demo_4o_mini.json")
sudo_workflow = None
# sudo_execution_result = "Sudo execution result for the given workflow."
sudo_execution_result = None
default_tools = []


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
        try:
            return OpenAILLMConfig(**llm_config_dict)
        except Exception:
            # If OpenAI config fails, try the base LLMConfig
            return LLMConfig(**llm_config_dict)


async def generate_task_info(goal: str, llm_config: Dict[str, Any], additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate detailed task information including workflow specifications from a goal.
    
    Args:
        goal: The high-level goal or objective
        llm_config: LLM configuration dictionary
        additional_info: Optional additional context or instructions
        
    Returns:
        Dictionary containing workflow specifications
    """
    try:
        # Create the task info prompt
        task_prompt = TASK_INFO_PROMPT.format(
            goal=goal,
            additional_info=additional_info or "No additional information provided."
        )
        
        # Initialize LLM
        llm_config_obj = create_llm_config(llm_config or default_llm_config)
        llm = create_llm_instance(llm_config_obj)
        
        # Generate task info
        response = llm.single_generate([{"role": "user", "content": task_prompt}])
        
        logger.info(f"Task info generation response: {response}")
        
        # Parse JSON from response
        task_info_list = parse_json_from_text(response)
        
        if not task_info_list:
            raise ValueError(f"No JSON found in LLM response: {response}")
        
        try:
            task_info = json.loads(task_info_list[0])
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Failed to parse: {task_info_list[0]}")
            raise ValueError(f"Invalid JSON in LLM response: {task_info_list[0]}")
        
        # Validate required fields
        required_fields = ["workflow_name", "workflow_description", "inputs_format", "outputs_format"]
        for field in required_fields:
            if field not in task_info:
                raise ValueError(f"Missing required field '{field}' in task info")
        
        logger.info(f"Successfully generated task info: {task_info}")
        return task_info
        
    except Exception as e:
        logger.error(f"Error generating task info: {str(e)}")
        raise


async def goal_based_workflow_generation(goal: str, llm_config: Dict[str, Any], additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a workflow from a high-level goal.
    
    This function:
    1. First converts the goal into detailed workflow specifications (task info)
    2. Then uses those specifications to generate the actual workflow
    
    Args:
        goal: The high-level goal or objective for the workflow
        llm_config: LLM configuration dictionary
        additional_info: Optional additional context or instructions
        
    Returns:
        Dictionary containing the generated workflow graph and metadata
    """
    try:
        logger.info(f"Starting workflow generation for goal: {goal}")
        
        # Step 1: Generate detailed task information from the goal
        logger.info("Step 1: Generating task information...")
        task_info = await generate_task_info(goal, llm_config, additional_info)
        
        # Check if we have a sudo workflow for testing
        if sudo_workflow:
            logger.info("Using sudo workflow for testing")
            
            # Use get_config() to get clean, serializable dictionary
            sudo_workflow_dict = sudo_workflow.get_config()
            
            return {
                "success": True,
                "workflow_graph": sudo_workflow_dict,  # Always return as dictionary
                "task_info": task_info,
                "original_goal": goal,
                "workflow_name": task_info["workflow_name"],
                "workflow_description": task_info["workflow_description"],
            }
        
        # Step 2: Create workflow generation prompt with task specifications
        logger.info("Step 2: Creating workflow generation prompt...")
        
        # Format inputs and outputs for the workflow generation prompt
        inputs_format_str = "\n".join([
            f"- **{key}**: {value}" 
            for key, value in task_info["inputs_format"].items()
        ])
        
        outputs_format_str = "\n".join([
            f"- **{key}**: {value}" 
            for key, value in task_info["outputs_format"].items()
        ])
        
        # Create the workflow generation prompt
        workflow_prompt = WORKFLOW_GENERATION_PROMPT.format(
            goal=goal,
            inputs_format=inputs_format_str,
            outputs_format=outputs_format_str
        )
        
        # Step 3: Generate the actual workflow using WorkFlowGenerator
        logger.info("Step 3: Generating workflow...")
        
        try:
            # Initialize LLM and WorkFlowGenerator
            llm_config_obj = create_llm_config(llm_config or default_llm_config)
            llm = create_llm_instance(llm_config_obj)
            
            # Create workflow generator
            workflow_generator = WorkFlowGenerator(llm=llm)
            
            # Generate the workflow using the detailed prompt
            workflow_graph = workflow_generator.generate_workflow(goal=workflow_prompt)
            
            logger.info("Successfully generated workflow")
            
            # Use get_config() to get clean, serializable dictionary
            workflow_graph_dict = workflow_graph.get_config()
            
            return {
                "success": True,
                "workflow_graph": workflow_graph_dict,  # Always return as dictionary
                "task_info": task_info,
                "original_goal": goal,
                "workflow_name": task_info["workflow_name"],
                "workflow_description": task_info["workflow_description"]
            }
            
        except Exception as e:
            logger.error(f"Error in workflow generation: {str(e)}")
            
            # Return task info with error
            return {
                "success": False,
                "error": f"Workflow generation failed: {str(e)}",
                "task_info": task_info,
                "original_goal": goal
            }
            
    except Exception as e:
        logger.error(f"Error in goal-based workflow generation: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate workflow: {str(e)}",
            "original_goal": goal
        }


async def workflow_execution(workflow_graph, llm_config: Dict[str, Any], inputs: Dict[str, Any], mcp_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a workflow graph with the given configuration.
    
    Args:
        workflow_graph: The workflow graph object (WorkFlowGraph) or dict to execute
        llm_config: LLM configuration dictionary
        inputs: Input parameters for the workflow execution
        mcp_config: Optional MCP configuration dictionary
        
    Returns:
        Dict containing execution results and status
    """
    try:
        logger.info("Starting workflow execution...")
        
        # Check if we have a sudo execution result for testing
        if sudo_execution_result:
            logger.info("Using sudo execution result for testing")
            return {
                "success": True,
                "status": "completed",
                "message": sudo_execution_result,
                "workflow_received": bool(workflow_graph),
                "llm_config_received": bool(llm_config),
                "mcp_config_received": bool(mcp_config),
                "inputs": inputs,
            }
        
        # Create LLM config and instance
        llm_config_obj = create_llm_config(llm_config or default_llm_config)
        llm = create_llm_instance(llm_config_obj)
        
        # Convert dictionary to WorkFlowGraph using from_dict()
        # The workflow_graph parameter should always be a dictionary when coming from API
        if isinstance(workflow_graph, dict):
            workflow_graph_obj = WorkFlowGraph.from_dict(workflow_graph)
        else:
            # This shouldn't happen in normal API flow, but handle gracefully
            logger.warning("Received non-dict workflow_graph, assuming it's already a WorkFlowGraph object")
            workflow_graph_obj = workflow_graph
        
        # Setup tools
        if mcp_config:
            mcp_toolkit = MCPToolkit(config=mcp_config)
            tools = mcp_toolkit.get_tools()
        else:
            # Use default tools (similar to server)
            try:
                tools = default_tools
            except Exception as e:
                logger.warning(f"Failed to load default tools: {e}, using empty tools list")
                tools = []
        
        # Create agent manager and add agents from workflow
        agent_manager = AgentManager(tools=tools)
        agent_manager.add_agents_from_workflow(workflow_graph_obj, llm_config=llm_config_obj)
        
        # Create and initialize workflow
        workflow = WorkFlow(graph=workflow_graph_obj, agent_manager=agent_manager, llm=llm)
        workflow.init_module()
        
        # Execute the workflow
        logger.info("Executing workflow with inputs...")
        output = await workflow.async_execute(inputs=inputs)
        
        logger.info("Workflow execution completed successfully")
        
        return {
            "success": True,
            "status": "completed",
            "message": output,
            "workflow_received": bool(workflow_graph),
            "llm_config_received": bool(llm_config),
            "mcp_config_received": bool(mcp_config),
            "inputs": inputs
        }
        
    except Exception as e:
        logger.error(f"Error in workflow execution: {str(e)}")
        return {
            "success": False,
            "status": "error",
            "message": f"Workflow execution failed: {str(e)}",
            "error": str(e),
            "workflow_received": bool(workflow_graph),
            "llm_config_received": bool(llm_config),
            "mcp_config_received": bool(mcp_config),
            "inputs": inputs
        }