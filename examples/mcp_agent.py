## You will need to create a MCP config file first at "examples/output/tests/shares_mcp.config"

import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate 
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.tools.image_analysis import ImageAnalysisTool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION_ID = os.getenv("OPENAI_ORGANIZATION_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)

def test_MCP_server():
    
    mcp_Toolkit = MCPToolkit(config_path="examples/output/mcp_agent/mcp.config")
    tools = mcp_Toolkit.get_toolkits()
    
    mcp_agent = CustomizeAgent(
        name="MCPAgent",
        description="A MCP agent that can use the tools provided by the MCP server",
        prompt_template= StringTemplate(
            instruction="Do some operations based on the user's instruction."
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "instruction", "type": "string", "description": "The goal you need to achieve"}
        ],
        outputs=[
            {"name": "result", "type": "string", "description": "The result of the operation"}
        ],
        tools=tools
    )
    mcp_agent.save_module("examples/output/mcp_agent/mcp_agent.json")
    mcp_agent.load_module("examples/output/mcp_agent/mcp_agent.json", llm_config=openai_config, tools=tools)

    message = mcp_agent(
        inputs={"instruction": "Summarize all the tools."}
    )
    
    print(f"Response from {mcp_agent.name}:")
    print(message.content.result)

def test_image_analysis_tool():
    tools = []
    tools.append(ImageAnalysisTool(api_key=OPENROUTER_API_KEY, model="openai/gpt-4o-mini"))
    
    mcp_agent = CustomizeAgent(
        name="MCPAgent",
        description="A MCP agent that can use the tools provided by the MCP server",
        prompt_template= StringTemplate(
            instruction="Do some operations based on the user's instruction."
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "prompt", "type": "string", "description": "The question or instruction for image analysis."},
            {"name": "image_url", "type": "string", "description": "The URL of the image to analyze."}
        ],
        outputs=[
            {"name": "content", "type": "string", "description": "The analysis result."},
            {"name": "usage", "type": "object", "description": "Token usage info."}
        ],
        tools=tools
    )
    mcp_agent.save_module("examples/output/mcp_agent/mcp_agent.json")
    mcp_agent.load_module("examples/output/mcp_agent/mcp_agent.json", llm_config=openai_config, tools=tools)

    message = mcp_agent(
        inputs={
            "prompt": "Describe this image.",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        }
    )
    print(f"Response from {mcp_agent.name}:")
    print(message.content.content)

def test_image_generation_tool():
    from evoagentx.tools.image_generation import ImageGenerationTool
    tools = []
    tools.append(ImageGenerationTool(api_key=OPENAI_API_KEY, organization_id=OPENAI_ORGANIZATION_ID, model="gpt-4o", save_path="./imgs"))

    mcp_agent = CustomizeAgent(
        name="MCPAgent",
        description="A MCP agent that can use the tools provided by the MCP server",
        prompt_template= StringTemplate(
            instruction="Do some operations based on the user's instruction."
        ), 
        llm_config=openai_config,
        inputs=[
            {"name": "prompt", "type": "string", "description": "The prompt for image generation."},
        ],
        outputs=[
            {"name": "file_path", "type": "object", "description": "The generated image (PIL.Image)."}
        ],
        tools=tools
    )
    mcp_agent.save_module("examples/output/mcp_agent/mcp_agent.json")
    mcp_agent.load_module("examples/output/mcp_agent/mcp_agent.json", llm_config=openai_config, tools=tools)

    message = mcp_agent(
        inputs={
            "prompt": "画一只像素风格的灰色虎斑猫抱着水獭",
        }
    )
    from PIL import Image
    img = Image.open(message.content.file_path)
    img.show()

if __name__ == "__main__":
    # test_MCP_server()
    # test_image_analysis_tool()
    test_image_generation_tool()