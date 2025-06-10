import os 
from dotenv import load_dotenv
from evoagentx.core import Message 
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate 
from evoagentx.core.module_utils import extract_code_blocks as util_extract_code_blocks
from evoagentx.core.registry import register_parse_function
from evoagentx.tools import FileTool 
from evoagentx.tools.mcp import MCPToolkit

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)


@register_parse_function
def extract_code_blocks(content: str) -> dict:
    return {"code": util_extract_code_blocks(content)[0]}

def build_customize_agent():

    # Create an agent from a dictionary
    agent_data = {
        "name": "FirstAgent",
        "description": "A simple agent that prints hello world",
        "prompt": "Print 'hello world'", 
        "llm_config": openai_config
    }
    agent = CustomizeAgent.from_dict(agent_data)

    # Execute the agent
    message: Message = agent() # the output of an agent is a Message object 

    print(f"Response from {agent.name}:")
    print(message.content.content) # the content of a Message object is a LLMOutputParser object


def build_customize_agent_with_inputs():

    simple_agent = CustomizeAgent(
        name="SimpleAgent",
        description="A basic agent that responds to queries",
        prompt="Answer the following question: {question}",
        llm_config=openai_config,
        inputs=[
            {"name": "question", "type": "string", "description": "The question to answer"}
        ]
    )
    # Execute the agent
    response = simple_agent(inputs={"question": "What is a language model?"})
    print(f"Response from {simple_agent.name}:")
    print(response.content.content)  # Access the raw response content 


def build_customize_agent_with_inputs_and_outputs():

    code_writer = CustomizeAgent(
        name="CodeWriter",
        description="Writes Python code based on requirements",
        prompt="Write Python code that implements the following requirement: {requirement}",
        llm_config=openai_config,
        inputs=[
            {"name": "requirement", "type": "string", "description": "The coding requirement"}
        ],
        outputs=[
            {"name": "code", "type": "string", "description": "The generated Python code"}
        ],
        parse_mode="str" # use the raw LLM output as the value for each output field
    )

    message = code_writer(
        inputs={"requirement": "Write a function that returns the sum of two numbers"}
    )
    print(f"Response from {code_writer.name}:")
    print(message.content.code)


def build_customize_agent_with_custom_parse_func():

    code_writer = CustomizeAgent(
        name="CodeWriter",
        description="Writes Python code based on requirements",
        prompt="Write Python code that implements the following requirement: {requirement}",
        llm_config=openai_config,
        inputs=[
            {"name": "requirement", "type": "string", "description": "The coding requirement"}
        ],
        outputs=[
            {"name": "code", "type": "string", "description": "The generated Python code"}
        ],
        parse_mode="custom", 
        parse_func=lambda content: {"code": util_extract_code_blocks(content)[0]}  # Extract first code block
    )

    message = code_writer(
        inputs={"requirement": "Write a function that returns the sum of two numbers"}
    )
    print(f"Response from {code_writer.name}:")
    print(message.content.code)


def build_customize_agent_with_prompt_template():

    agent = CustomizeAgent(
        name="FirstAgent",
        description="A simple agent that prints hello world",
        prompt_template=StringTemplate(
            instruction="Print 'hello world'",
        ),
        llm_config=openai_config
    )

    message = agent()
    print(f"Response from {agent.name}:")
    print(message.content.content)


def build_customize_agent_with_inputs_and_outputs_and_prompt_template(): 

    code_writer = CustomizeAgent(
        name="CodeWriter",
        description="Writes Python code based on requirements",
        prompt_template= StringTemplate(
            instruction="Write Python code that implements the provided `requirement`",
            # demonstrations=[
            #     {
            #         "requirement": "print 'hello world'",
            #         "code": "print('hello world')"
            #     }, 
            #     {
            #         "requirement": "print 'Test Demonstration'",
            #         "code": "print('Test Demonstration')"
            #     }
            # ]
        ), # no need to specify input placeholders in the instruction of the prompt template
        llm_config=openai_config,
        inputs=[
            {"name": "requirement", "type": "string", "description": "The coding requirement"}
        ],
        outputs=[
            {"name": "code", "type": "string", "description": "The generated Python code"},
        ],
        parse_mode="custom", 
        parse_func=lambda content: {"code": util_extract_code_blocks(content)[0]}
    )

    message = code_writer(
        inputs={"requirement": "Write a function that returns the sum of two numbers"}
    )
    print(f"Response from {code_writer.name}:")
    print(message.content.code)


def build_customize_agent_with_tools():

    code_writer = CustomizeAgent(
        name="CodeWriter",
        description="Writes Python code based on requirements",
        prompt_template= StringTemplate(
            instruction="Write Python code that implements the provided `requirement` and save the code to the provided `file_path`"
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "requirement", "type": "string", "description": "The coding requirement"},
            {"name": "file_path", "type": "string", "description": "The path to save the code"}
        ],
        # outputs=[
        #     {"name": "code", "type": "string", "description": "The generated Python code"}
        # ],
        tool_names=["file_tool"],
        tool_dict={"file_tool": FileTool()}
    )

    message = code_writer(
        inputs={"requirement": "Write a function that returns the sum of two numbers", "file_path": "output/test_code.py"}
    )
    # print(f"Response from {code_writer.name}:")
    # print(message.content.code)

def build_customize_agent_with_MCP(config_path):
    mcp_toolkit = MCPToolkit(config_path=config_path)
    tools = mcp_toolkit.get_tools()
    
    tools_mapping = {}
    tools_schemas = [(tool.get_tool_schemas(), tool) for tool in tools]
    tools_schemas = [(j, k) for i, k in tools_schemas for j in i]
    for tool_schema, tool in tools_schemas:
        tool_name = tool_schema["function"]["name"]
        tools_mapping[tool_name] = tool
    tool_names=[tool_schema["function"]["name"] for tool_schema, _ in tools_schemas]

    customize_agent = CustomizeAgent(
        name="MCPToolUser",
        description="Do some tasks using the tools",
        prompt_template= StringTemplate(
            instruction="Do some tasks using the tools"
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "instruction", "type": "string", "description": "The instruction to the tool user"}
        ],
        outputs=[
            {"name": "result", "type": "string", "description": "The result of the task"},
            {"name": "tool_calls", "type": "string", "description": "The tool calls used to get the result (if any)"}
        ],
        tool_names=tool_names,
        tool_dict=tools_mapping
    )

    message = customize_agent(
        inputs={"instruction": "Summarize all your tools."}
    )
    print(f"Response from {customize_agent.name}:")
    print(message.content)

if __name__ == "__main__":
    build_customize_agent()
    build_customize_agent_with_inputs()
    build_customize_agent_with_inputs_and_outputs()
    build_customize_agent_with_custom_parse_func()
    build_customize_agent_with_prompt_template()
    build_customize_agent_with_inputs_and_outputs_and_prompt_template()
    build_customize_agent_with_tools()
    
    
    config_path = "examples/output/tests/shares_mcp.config"
    if os.path.exists(config_path):
        build_customize_agent_with_MCP(config_path=config_path)
    else:
        print(f"You will need to provide a MCP config file at {config_path} to test.")