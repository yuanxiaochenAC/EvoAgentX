"""
Required servers: Hirebase, PDF Reader
Links: 
- https://github.com/jhgaylor/hirebase-mcp
- https://github.com/sylphlab/pdf-reader-mcp
"""

import asyncio
import os
from dotenv import load_dotenv

from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.tools.mcp import MCPToolkit, MCPClient
from evoagentx.core.logging import logger
from evoagentx.agents.cus_tool_caller import CusToolCaller
from evoagentx.prompts.tool_caller import TOOL_CALLER_PROMPT
from evoagentx.core.message import Message, MessageType
load_dotenv()

def main():
    logger.info("=== Custom Tool Caller Agent with MCP Tools Example ===")
    
    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Initialize the language model
    llm_config = OpenAILLMConfig(
        llm_type="OpenAILLM",
        model="gpt-4o-mini",
        openai_key=api_key,
        temperature=0.3,
    )
    
    pdf_file_path = "./examples/test_pdf.pdf"
    config_path = "./examples/mcp.config"
    
    try:
        # Create MCP toolkit separately and add it to the agent
        logger.info("\nCreating Tool Caller Agent with separate MCP toolkit...")
        
        # Define the inputs and outputs for the agent
        inputs = [
            {"name": "query", "type": "str", "description": "The query from the user"}
        ]
        
        outputs = [
            {"name": "answer", "type": "str", "description": "The answer to the query"}
        ]
        
        mcp_toolkit = MCPToolkit(config_path=config_path)
        tools = mcp_toolkit.get_tools()
        # import pdb; pdb.set_trace()
        # print(tools[1].tools[0](**{"query": "camel-ai"}))
        
        
        print("Creating Tool Caller Agent...")
        tool_caller = CusToolCaller(name="Tool Caller", 
                                    description="Tool Caller Agent", 
                                    llm_config=llm_config, 
                                    mcp_config_path=config_path,
                                    prompt=TOOL_CALLER_PROMPT["system_prompt"],
                                    inputs=inputs,
                                    outputs=outputs)
        tool_caller.add_tools(tools)
        
        print("Executing Tool Caller Agent...")
        message = Message(content="Give a short summary of the github project camel-ai", agent="user", msg_type=MessageType.REQUEST)
        message_out = tool_caller.execute(action_name=tool_caller.tool_calling_action_name,
                                                msgs=message,
                                                return_msg_type=MessageType.RESPONSE)
        print(message_out)
       
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()