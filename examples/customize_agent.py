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
    print(f"Response from {code_writer.name}:")
    print(message.content)

def build_customize_agent_with_MCP(config_path):
    mcp_toolkit = MCPToolkit(config_path=config_path)
    tools = mcp_toolkit.get_tools()
    
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
        tool_names=[tool.name for tool in tools],
        tool_dict={tool.name: tool for tool in tools}
    )

    message = customize_agent(
        inputs={"instruction": "Summarize all your tools."}
    )
    print(f"Response from {customize_agent.name}:")
    print(message.content)

def build_customize_agent_with_custom_parse_and_format():
    """Test case demonstrating custom parse function and output format with XML."""
    
    def custom_xml_parser(content: str) -> dict:
        """Custom parser that extracts data from XML-like format."""
        # Simple XML-style parsing (for demonstration)
        result = {}
        for field in ["name", "age", "occupation"]:
            start_tag = f"<{field}>"
            end_tag = f"</{field}>"
            try:
                start_idx = content.index(start_tag) + len(start_tag)
                end_idx = content.index(end_tag)
                result[field] = content[start_idx:end_idx].strip()
            except ValueError:
                result[field] = ""
        return result

    person_info_agent = CustomizeAgent(
        name="PersonInfoExtractor",
        description="Extracts structured person information in XML format",
        prompt="Extract information about the following person: {person_description}",
        llm_config=openai_config,
        inputs=[
            {"name": "person_description", "type": "string", "description": "Description of the person"}
        ],
        outputs=[
            {"name": "name", "type": "string", "description": "Person's name"},
            {"name": "age", "type": "string", "description": "Person's age"},
            {"name": "occupation", "type": "string", "description": "Person's occupation"}
        ],
        parse_mode="custom",
        parse_func=custom_xml_parser,
        custom_output_format="""Please format your response in XML tags:
<name>person's name</name>
<age>person's age</age>
<occupation>person's occupation</occupation>"""
    )

    message = person_info_agent(
        inputs={"person_description": "John is a 35-year-old software engineer who loves coding."}
    )
    print(f"Response from {person_info_agent.name}:")
    print("Name:", message.content.name)
    print("Age:", message.content.age)
    print("Occupation:", message.content.occupation)

def build_customize_agent_with_json_parse():
    """Test case demonstrating JSON parse mode for structured data extraction."""
    
    recipe_analyzer = CustomizeAgent(
        name="RecipeAnalyzer",
        description="Analyzes recipe information and returns structured data",
        prompt="""Analyze the following recipe and extract key information.
Recipe: {recipe_text}

Please format your response as a JSON object with the following structure:
{
    "name": "Recipe name",
    "prep_time_minutes": number,
    "ingredients": ["ingredient1", "ingredient2", ...],
    "difficulty": "easy|medium|hard"
}""",
        llm_config=openai_config,
        inputs=[
            {"name": "recipe_text", "type": "string", "description": "The recipe text to analyze"}
        ],
        outputs=[
            {"name": "name", "type": "string", "description": "Name of the recipe"},
            {"name": "prep_time_minutes", "type": "int", "description": "Preparation time in minutes"},
            {"name": "ingredients", "type": "list", "description": "List of ingredients"},
            {"name": "difficulty", "type": "string", "description": "Difficulty level of the recipe"}
        ],
        parse_mode="json"  # This will automatically parse JSON response into structured output
    )

    sample_recipe = """
    Classic Chocolate Chip Cookies
    
    Mix 2 1/4 cups flour, 1 cup butter, 3/4 cup sugar, 2 eggs, 
    1 tsp vanilla extract, and 2 cups chocolate chips. 
    Bake at 375°F for 10-12 minutes.
    Total prep time: 25 minutes.
    """

    message = recipe_analyzer(inputs={"recipe_text": sample_recipe})
    print(f"\nResponse from {recipe_analyzer.name}:")
    print("Recipe Name:", message.content.name)
    print("Prep Time:", message.content.prep_time_minutes, "minutes")
    print("Ingredients:", ", ".join(message.content.ingredients))
    print("Difficulty:", message.content.difficulty)

if __name__ == "__main__":
    build_customize_agent()
    build_customize_agent_with_inputs()
    build_customize_agent_with_inputs_and_outputs()
    build_customize_agent_with_custom_parse_func()
    build_customize_agent_with_prompt_template()
    build_customize_agent_with_inputs_and_outputs_and_prompt_template()
    build_customize_agent_with_tools()
    build_customize_agent_with_custom_parse_and_format()
    build_customize_agent_with_json_parse()
    
    config_path = "examples/output/tests/shares_mcp.config"
    if os.path.exists(config_path):
        build_customize_agent_with_MCP(config_path=config_path)
    else:
        print(f"You will need to provide a MCP config file at {config_path} to test.")