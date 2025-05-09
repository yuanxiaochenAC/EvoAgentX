from typing import Dict, Any, Optional, Callable
import asyncio
import json
from .customize_agent import CustomizeAgent, OUTPUT_EXTRACTION_PROMPT
from ..actions.tool_calling import ToolCalling
from ..prompts.tool_caller import CUSTOM_TOOL_CALLER_PROMPT
from ..core.message import Message, MessageType
from ..core.module_utils import parse_json_from_llm_output
from ..models.base_model import LLMOutputParser
from ..tools.tool import Tool

class CusToolCaller(CustomizeAgent):
    """
    An agent that determines whether to call a tool or provide a direct answer
    based on analyzing the user query and available tools.
    
    This agent has two main actions:
    1. ToolCalling - Executes the selected tool if required
    """
    tools_schema: Optional[Dict[str, dict]] = None
    tools_caller: Optional[Dict[str, Callable]] = None
    tool_calling_action: Optional[ToolCalling] = None
    _initialized: bool = False
    ori_prompt: str = ""
    tool_calling_prompt: str = ""
    
    def __init__(self, **kwargs):
        # Extract our specific parameters before passing to parent
        max_tool_try = kwargs.pop("max_tool_try", 2)
        # Create actions with the max_tool_try value
        # Don't add to kwargs - let CustomizeAgent handle action creation
        tool_actions = [ToolCalling(max_tool_try=max_tool_try)]
        tools = kwargs.get("tools", None)
        
        # Initialize as CustomizeAgent - all other parameters should be passed by the caller
        super().__init__(**kwargs)
        self.outputs_format = self.actions[0].outputs_format
        self.actions = []
        self._action_map = {}
        self.init_module()
        self.tools = tools
        
        # Now add our tool actions (after CustomizeAgent has created its action)
        self.ori_prompt = self.system_prompt + kwargs.get("prompt", "")
        for action in tool_actions:
            self.add_action(action)
        
        # Store values as instance attributes after parent initialization
        self.max_tool_try = max_tool_try
        # Store the tool calling action for easy access
        self.tool_calling_action = self._action_map[self.tool_calling_action_name]
        self.add_tools(tools)
        self.tool_calling_prompt = self.ori_prompt + CUSTOM_TOOL_CALLER_PROMPT + "\n### Tools Available\n" + str(self.tools_schema)
        
    @property
    def tool_calling_action_name(self) -> str:  
        return self.get_action_name(action_cls=ToolCalling)
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the tool caller agent"""
        self.add_tools([tool])
    
    def add_tools(self, tools: list[Tool]):
        tools_schemas = [tool.get_tool_schemas() for tool in tools]
        tools_schemas = [j for i in tools_schemas for j in i]
        tools_callers = [tool.get_tools() for tool in tools]
        tools_callers = [j for i in tools_callers for j in i]
        tools_names = [i["function"]["name"] for i in tools_schemas]
        if not self.tools_schema:
            self.tools_schema = {}
            self.tools_caller = {}
        for tool_schema, tool_caller, tool_name in zip(tools_schemas, tools_callers, tools_names):
            self.tools_schema[tool_name] = tool_schema
            self.tools_caller[tool_name] = tool_caller
        
        self.tool_calling_action.add_tools(tools)
        self._initialized = True
    
    def execute(self, **kwargs) -> Message:
        """Execute the tool caller agent"""
        try:
            # # Always use the tool_calling_action for CusToolCaller, override the action_name
            action_name = kwargs.get("action_name", self.tool_calling_action_name)
            if action_name == self.tool_calling_action_name:
                self.system_prompt = self.tool_calling_prompt
                action_input_data = kwargs.get("action_input_data", {})
                if not action_input_data.get("query") and "query" not in action_input_data:
                    # If no query is provided, create one from the task description or goal
                    goal = kwargs.get("wf_goal", "")
                    task_desc = kwargs.get("wf_task_desc", "")
                    if task_desc:
                        action_input_data["query"] = f"Task: {task_desc}\nGoal: {goal}"
                    elif goal:
                        action_input_data["query"] = goal
                    
                    # action_input_data["query"] += "\n\n current query/goal and config: \n" + json.dumps(kwargs.get("action_input_data", {}), indent=4)
                kwargs["action_input_data"] = action_input_data
                
                additional_info = Message(
                    content=json.dumps(kwargs.get("action_input_data", {}), indent=4),
                    agent=self.name,
                    action=action_name,
                    msg_type=MessageType.RESPONSE,
                    wf_goal=kwargs.get("wf_goal", ""),
                    wf_task=kwargs.get("wf_task", ""),
                    wf_task_desc=kwargs.get("wf_task_desc", "")
                )
                kwargs["history"] = [additional_info]
                
            else:
                self.system_prompt = self.ori_prompt
            
            # from pdb import set_trace; set_trace()
            
            # # Execute using the parent's execute_async method
            llm_output = super().execute(**kwargs)
            print(f"Agent {self.name} execution completed successfully")
            
            
            if action_name != self.tool_calling_action_name:
                return llm_output
            
            print("_____________________ Start Extracting Output _____________________")
            attr_descriptions: dict = self.outputs_format.get_attr_descriptions()
            output_description_list = [] 
            for i, (name, desc) in enumerate(attr_descriptions.items()):
                output_description_list.append(f"{i+1}. {name}\nDescription: {desc}")
            output_description = "\n\n".join(output_description_list)
            extraction_prompt = self.ori_prompt + "\n\n" + OUTPUT_EXTRACTION_PROMPT.format(text=llm_output.content, output_description=output_description)
            llm_extracted_output: LLMOutputParser = self.llm.generate(prompt=extraction_prompt, history=kwargs.get("history", []) + [llm_output])
            llm_extracted_data: dict = parse_json_from_llm_output(llm_extracted_output.content)
            output = self.outputs_format.from_dict(llm_extracted_data)
            
            # Create a proper Message with the output
            return_msg_type = kwargs.get("return_msg_type", MessageType.RESPONSE)
            
            # Create a message with the extracted output as content
            message = Message(
                content=output,
                agent=self.name,
                action=action_name,
                msg_type=return_msg_type,
                wf_goal=kwargs.get("wf_goal", ""),
                wf_task=kwargs.get("wf_task", ""),
                wf_task_desc=kwargs.get("wf_task_desc", "")
            )
            return message
            
        except Exception as e:
            print(f"Error in CusToolCaller.execute for agent {self.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create an error message to return
            error_message = Message(
                content=f"Error executing agent {self.name}: {str(e)}",
                agent=self.name,
                action=kwargs.get("action_name", self.tool_calling_action_name),
                msg_type=MessageType.ERROR
            )
            return error_message


