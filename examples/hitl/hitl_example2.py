import asyncio
from dotenv import load_dotenv
import os
from typing import Optional
from pydantic import Field
from evoagentx.models.base_model import BaseLLM
from evoagentx.workflow import WorkFlow, WorkFlowGraph
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
from evoagentx.agents import Agent, CustomizeAgent, AgentManager
from evoagentx.actions import Action, ActionInput, ActionOutput
from evoagentx.hitl import (
    HITLUserInputCollectorAgent,
    HITLManager
)
from evoagentx.models import OpenAILLMConfig, OpenAILLM 

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class UserProfileInput(ActionInput):
    user_name: str = Field(description="User's name")
    user_age: int = Field(description="User's age") 
    user_email: str = Field(description="User's email address")
    user_preferences: Optional[str] = Field(default=None, description="User's preferences")

class UserProfileOutput(ActionOutput):
    profile_summary: str = Field(description="User's profile summary")
    recommendations: str = Field(description="Personalized recommendations based on user information")

# class ProfileProcessingAction(Action):
#     def __init__(
#         self,
#         name: str="ProfileProcessingAction",
#         description: str="Process user's profile and generate recommendations",
#         prompt: str="Generate profile summary and personalized recommendations based on the following user information:\nName: {user_name}\nAge: {user_age}\nEmail: {user_email}\nPreferences: {user_preferences}\n\nPlease provide profile summary and personalized recommendations.",
#         inputs_format: ActionInput=None,
#         outputs_format: ActionOutput=None,
#         **kwargs
#     ):
#         inputs_format = inputs_format or UserProfileInput
#         outputs_format = outputs_format or UserProfileOutput
#         super().__init__(
#             name=name,
#             description=description,
#             prompt=prompt,
#             inputs_format=inputs_format,
#             outputs_format=outputs_format,
#             **kwargs
#         )
    
#     def execute(
#         self, 
#         llm: Optional[BaseLLM] = None, 
#         inputs: Optional[dict] = None, 
#         sys_msg: Optional[str] = None, 
#         return_prompt: bool = False, 
#         **kwargs
#     ) -> UserProfileOutput:
#         action_input_attrs = self.inputs_format.get_attrs()
#         action_input_data = {attr: inputs.get(attr, "not provided") for attr in action_input_attrs}
#         prompt = self.prompt.format(**action_input_data)
        
#         output = llm.generate(
#             prompt=prompt, 
#             system_message=sys_msg or "You are a professional user profile analyst.",
#             parser=self.outputs_format, 
#             parse_mode="json"
#         )
#         if return_prompt:
#             return output, prompt
#         return output

#     async def async_execute(
#         self, 
#         llm: Optional[BaseLLM] = None, 
#         inputs: dict = None, 
#         sys_msg: str = None, 
#         return_prompt: bool = False, 
#         **kwargs
#     ) -> UserProfileOutput:
#         return self.execute(llm, inputs, sys_msg, return_prompt, **kwargs)

async def main():
    print("🚀 EvoAgentX HITL user input collection example")
    print("=" * 60)

    llm_config = OpenAILLMConfig(model="gpt-4o", openai_key=OPENAI_API_KEY, stream=True, output_response=True)
    llm = OpenAILLM(llm_config)

    # define user input fields
    user_input_fields = {
        "user_name": {
            "type": "string",
            "description": "please input your name",
            "required": True
        },
        "user_age": {
            "type": "int", 
            "description": "please input your age",
            "required": True
        },
        "user_email": {
            "type": "string",
            "description": "please input your email address",
            "required": True
        },
        "user_preferences": {
            "type": "string",
            "description": "please input your preferences (optional)",
            "required": False,
            "default": "no special preferences"
        }
    }

    # create user input collector agent
    user_input_collector = HITLUserInputCollectorAgent(
        name="UserProfile",
        # description="collect user's input",
        llm_config=llm_config,
        input_fields=user_input_fields,
    )

    # create profile processing agent
    # profile_processor = Agent(
    #     name="ProfileProcessor",
    #     description="process user's profile and generate recommendations",
    #     actions=[ProfileProcessingAction()],
    #     llm_config=llm_config,
    # )
    profile_processor = CustomizeAgent(
        name="ProfileProcessor",
        description="process user's profile and generate recommendations",
        inputs=[
            {"name": "user_name", "type": "string", "description": "user's name"},
            {"name": "user_age", "type": "string", "description": "user's age"},
            {"name": "user_email", "type": "string", "description": "user's email"},
            {"name": "user_preferences", "type": "string", "description": "user's preferences"}
        ],
        outputs=[
            {"name": "profile_summary", "type": "string", "description": "profile summary based on user's information"},
            {"name": "recommendations", "type": "string", "description": "Personalized recommendations based on user information"}
        ],
        prompt="Generate profile summary and personalized recommendations based on the following user information:\nName: {user_name}\nAge: {user_age}\nEmail: {user_email}\nPreferences: {user_preferences}\n\nPlease provide profile summary and personalized recommendations. The results should be presented in json format",
        llm_config=llm_config,
        parse_mode="json"
    )
    
    # activate HITL feature
    hitl_manager = HITLManager()
    hitl_manager.activate()
    
    # define workflow nodes
    nodes = [
        WorkFlowNode(
            name="user_input_collection_node",
            description="collect user's input",
            agents=[user_input_collector],
            inputs=[],  # no external input
            outputs=[
                {"name": "user_name", "type": "string", "description": "user's name"},
                {"name": "user_age", "type": "int", "description": "user's age"},
                {"name": "user_email", "type": "string", "description": "user's email"},
                {"name": "user_preferences", "type": "string", "description": "user's preferences"}
            ]
        ),
        WorkFlowNode(
            name="profile_processing_node",
            description="process user's profile and generate recommendations",
            agents=[profile_processor],
            inputs=[
                {"name": "user_name", "type": "string", "description": "user's name"},
                {"name": "user_age", "type": "int", "description": "user's age"},
                {"name": "user_email", "type": "string", "description": "user's email"},
                {"name": "user_preferences", "type": "string", "description": "user's preferences"}
            ],
            outputs=[
                {"name": "profile_summary", "type": "string", "description": "user's profile summary"},
                {"name": "recommendations", "type": "string", "description": "personalized recommendations"}
            ]
        )
    ]
    
    # define workflow edges
    edges = [
        WorkFlowEdge(source="user_input_collection_node", target="profile_processing_node")
    ]
    
    # create workflow graph
    graph = WorkFlowGraph(
        goal="collect user's input and generate personalized profile and recommendations",
        nodes=nodes,
        edges=edges
    )
    
    agents = [user_input_collector, profile_processor]

    # set input output mapping
    hitl_data_mapping = {
        "user_name": "user_name",
        "user_age": "user_age", 
        "user_email": "user_email",
        "user_preferences": "user_preferences"
    }
    hitl_manager.hitl_input_output_mapping = hitl_data_mapping
    
    # create agent manager and workflow
    agent_manager = AgentManager(agents=agents)
    workflow = WorkFlow(
        graph=graph, 
        llm=llm, 
        agent_manager=agent_manager,
        hitl_manager=hitl_manager
    )

    try:
        print("\nstart to execute workflow...")
        print("system will prompt you to input user information.")
        
        # execute workflow
        result = await workflow.async_execute(
            inputs={},  # no initial input
            task_name="user profile collection and analysis",
            goal="collect user's input and generate personalized profile and recommendations"
        )
    
        print("\n" + "="*60)
        print("🎉 workflow executed successfully!")
        print("="*60)
        print("final result:\n")
        print(result)
    except Exception as e:
        print(f"workflow execution failed: {e}")
    finally:
        # deactivate HITL feature
        hitl_manager.deactivate()

if __name__ == "__main__":
    asyncio.run(main()) 