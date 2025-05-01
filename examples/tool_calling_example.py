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
# from evoagentx.core.message import Message, MessageType
# from evoagentx.prompts.tool_caller import TOOL_CALLER_PROMPT
# from evoagentx.agents.cus_tool_caller import CusToolCaller
# from evoagentx.tools.search_wiki import SearchWiki

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
        print("Trying to call tool with direct parameters:")
        print(tools[1].tools[0](**{"query": "camel-ai"}))
        
        # # Create the CustomizeAgent (CusToolCaller) with all required parameters
        # tool_caller = CusToolCaller(
        #     name=TOOL_CALLER_PROMPT["name"],
        #     description=TOOL_CALLER_PROMPT["description"],
        #     system_prompt=TOOL_CALLER_PROMPT["system_prompt"],
        #     prompt=TOOL_CALLER_PROMPT["prompt"],
        #     inputs=inputs,
        #     outputs=outputs,
        #     llm_config=llm_config,
        #     max_tool_try=1,
        #     mcp_config_path=config_path
        # )
        # tool_caller.add_tools(SearchWiki().get_tool_schemas(), SearchWiki().get_tools())
        
        # # Add the MCP toolkit to the agent
        # logger.info("MCP toolkit added to the agent")
        
        # # Create a user message with a query
        # # pdf_query = f"Read and summarize the content from this PDF file: {pdf_file_path}"
        # pdf_query = f"Search Wikipedia for the article about 'Python'"
        # user_message = Message(
        #     content=pdf_query,
        #     agent="user",
        #     msg_type=MessageType.REQUEST
        # )
        
        # # Execute the tool calling action
        # logger.info("Executing PDF content extraction...")
        # response = await tool_caller.execute(
        #     action_name=tool_caller.tool_calling_action_name,
        #     msgs=[user_message]
        # )
        
        # # Print the response
        # logger.info("\n===== Response =====")
        # # Log the content of the response to debug
        # logger.info(f"Response content type: {type(response.content)}")
        
        # if isinstance(response.content, dict):
        #     # Dictionary format
        #     if "answer" in response.content:
        #         logger.info(f"Answer: {response.content['answer']}")
        #     else:
        #         logger.info(f"Content (dict): {response.content}")
        # else:
        #     # Object format with attributes
        #     logger.info(f"Content (object): {response.content}")
        
        # # Clean up resources
        # await tool_caller.cleanup()
        # logger.info("Resources cleaned up")
        
        # logger.info("\nExample completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()