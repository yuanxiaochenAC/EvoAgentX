import asyncio
import os
import json
import http.server
import socketserver
import threading
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from evoagentx.agents.tool_caller import ToolCaller
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.core.logging import logger
from evoagentx.core.message import Message, MessageType
load_dotenv()


async def main():
    logger.info("=== ToolCaller Agent with MCP Tools Example ===")
    
    # Load environment variables
    
    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Initialize the language model
    llm_config = OpenAILLMConfig(
        llm_type="OpenAILLM",
        model="gpt-4o-mini",  # Using GPT-4o-mini for tool usage capabilities
        openai_key=api_key,
        temperature=0.7,
        max_tokens=1000,
    )
    
    pdf_file_path = "/home/junhual1/projects/EvoAgentX/tests/test_pdf.pdf"
    
    # # Start a simple HTTP server to serve the PDF file
    # ori_dir = os.getcwd()
    # pdf_dir = Path(pdf_file_path).parent.absolute()
    # pdf_name = Path(pdf_file_path).name
    # os.chdir(pdf_dir)  # Change working directory to where the PDF is
    
    # # Use a random available port
    # with socketserver.TCPServer(("", 0), http.server.SimpleHTTPRequestHandler) as httpd:
    #     port = httpd.server_address[1]
    #     # Start the server in a separate thread
    #     server_thread = threading.Thread(target=httpd.serve_forever)
    #     server_thread.daemon = True  # So it automatically terminates when main program ends
    #     server_thread.start()
        
    #     pdf_url = f"http://localhost:{port}/{pdf_name}"
    #     logger.info(f"Serving PDF at: {pdf_url}")
    # os.chdir(ori_dir)
    # Create MCP client and toolkit
    config_path = "tests/mcp.config"
    toolkit = MCPToolkit(config_path=config_path)
    
    
    try:
        # Connect to the MCP server
        await toolkit.connect()
        logger.info(f"Connected to MCP server: {toolkit.is_connected()}")
        
        # test_args = {
        #     "sources": [
        #             {
        #                 # "url": "http://localhost:45110/test_pdf.pdf"
        #                 "path": "/tests/test_pdf.pdf"
        #             }
        #         ],
        #         "include_full_text": True,
        #         "include_metadata": True,
        #         "include_page_count": True
        # }
        
        # print(toolkit.get_all_openai_tool_schemas())
        
        # result = await toolkit.call_tool(
        #     tool_name="read_pdf",
        #     tool_args=test_args
        # )
        
        # print(result)
        
        # assert False
        
        
        # Create ToolCaller agent
        agent = ToolCaller(llm_config=llm_config)
        agent.add_mcp_toolkit(toolkit)
        
        # Create a user message to process
        logger.info("Creating user message for the ToolCaller agent")
        
        # Test query for the agent
        # user_query = f"Can you summarize the pdf file at {pdf_url}"
        user_query = f"Can you summarize the pdf file at {pdf_file_path}"
        
        # Create a direct tool call without using the generator
        # This avoids the parsing issues
        # search_query = [{"function_name": "search_repositories", "function_args": {"query": "camel-ai/camel"}}]
        
        input_message = Message(content=user_query, msg_type=MessageType.REQUEST, agent="user")
        
        message_out = await agent.execute(
            action_name=agent.tool_calling_action_name,
            msgs=input_message,
            return_msg_type=MessageType.RESPONSE,
        )
        print(f"Output: {message_out}")
        
        message_out = await agent.execute(
            action_name=agent.tool_summarizing_action_name,
            msgs=message_out,
            return_msg_type=MessageType.RESPONSE,
        )
        print(f"Output: {message_out}")
        
        # message_out = await agent.execute(
        #     action_name=agent.tool_calling_action_name,
        #     msgs=message_out,
        #     return_msg_type=MessageType.RESPONSE,
        # )
        
        # print(f"Output: {message_out}")
        
            
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Clean up resources
        await toolkit.disconnect()
        logger.info("Disconnected from MCP server")
    
    logger.info("\nExample completed!")

if __name__ == "__main__":
    asyncio.run(main())