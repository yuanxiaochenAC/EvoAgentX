#!/usr/bin/env python
"""
Example demonstrating how to use EvoAgentX MCP tool.

This example shows how to:
1. Connect to an MCP server
2. List available tools
3. Call an MCP tool
4. Use the toolkit to manage multiple MCP servers

To run this example:
```
python examples/mcp_example.py
```

Note: You need to have an MCP server running or accessible before using this example.

Required servers: GitHub
Links: https://github.com/modelcontextprotocol/servers/tree/main/src/github
"""

import asyncio
import json
import os
import sys
import signal
import traceback
import dotenv
from evoagentx.tools import MCPToolkit
from evoagentx.core.logging import logger
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.agents.tool_caller import ToolCaller
from evoagentx.core.message import Message, MessageType
from evoagentx.actions.tool_calling import SUMMARIZING_PROMPT
dotenv.load_dotenv()

# Set up signal handlers for graceful shutdown
def handle_exit_signal(signum, frame):
    """Handle exit signals to cleanly terminate the process"""
    print(f"\nReceived signal {signum}, shutting down...")
    # Force exit without running any more async code
    os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

# For Windows, we need a different approach
if os.name == 'nt':
    try:
        import win32api
        def windows_handler(sig):
            handle_exit_signal(0, None)
        win32api.SetConsoleCtrlHandler(windows_handler, True)
    except ImportError:
        pass

# Example for multiple MCP servers
async def main():
    # Run the examples
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
    )
    
    config_path = "examples/mcp.config"
    toolkit = MCPToolkit(config_path=config_path)
    
    try:
        # Connect to the MCP server
        await toolkit.connect()
        print(toolkit.get_all_openai_tool_schemas())
        logger.info(f"Connected to MCP server: {toolkit.is_connected()}")
        print(toolkit.get_all_openai_tool_schemas())
        
        # Create ToolCaller agent
        tool_caller_agent = ToolCaller(llm_config=llm_config, max_tool_try=2)
        tool_caller_agent.add_mcp_toolkit(toolkit)
        
        # Create a user message to process
        logger.info("Creating user message for the ToolCaller agent")
        
        ## ___________ PDF Summarizer Agent ___________
        project_name = "camel-ai/owl"
        user_query = f"""
        Assume you are a tech lead and is looking for open projects that might be useful for your task.
        We are currently looking into {project_name} project. 
        Please look into this project and provide a comprehensive summary.
        """
        input_message = Message(content=user_query, msg_type=MessageType.REQUEST, agent="user")
        
        message_out = await tool_caller_agent.execute(
            action_name=tool_caller_agent.tool_calling_action_name,
            msgs=input_message,
            return_msg_type=MessageType.RESPONSE,
        )
        # print(f"Searching output: {message_out}")
        
        message_out_summarizing = await tool_caller_agent.execute(
            action_name=tool_caller_agent.tool_summarizing_action_name,
            msgs=input_message,
            history=[message_out],
            sys_msg=SUMMARIZING_PROMPT,
            return_msg_type=MessageType.RESPONSE
        )
        print(f"Summarizing output: {message_out_summarizing}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Clean up resources
        print("Disconnecting from MCP server")
        try:
            # Set a short timeout for disconnect to avoid hanging
            disconnect_task = asyncio.create_task(toolkit.disconnect())
            try:
                # Wait for disconnect with a timeout
                await asyncio.wait_for(disconnect_task, timeout=2.0)
                print("Successfully disconnected from MCP server")
            except asyncio.TimeoutError:
                print("Disconnect timed out, forcing termination")
                # Force cancel the task if it's taking too long
                disconnect_task.cancel()
                
                # Give a moment for cancellation to propagate
                await asyncio.sleep(0.1)
                
                # Force kill any servers that might be hanging by getting the current event loop and stopping it
                loop = asyncio.get_running_loop()
                for task in asyncio.all_tasks(loop=loop):
                    if task is not asyncio.current_task():
                        task.cancel()
                
                # Try to terminate subprocesses directly (works on both Windows and Unix)
                for server in toolkit.servers:
                    try:
                        # Access internal process objects
                        if hasattr(server, '_process') and server._process:
                            # Try to kill the process directly
                            server._process.kill()
                            print(f"Killed server process")
                        # Check for stdio client with subprocess
                        elif hasattr(server, 'exit_stack') and hasattr(server.exit_stack, '_exit_callbacks'):
                            # Try to find and kill any subprocesses in the exit callbacks
                            for callback in server.exit_stack._exit_callbacks:
                                if hasattr(callback, '__self__') and hasattr(callback.__self__, '_process'):
                                    callback.__self__._process.kill()
                                    print(f"Killed subprocess from exit stack")
                    except Exception as kill_error:
                        print(f"Error killing subprocess: {kill_error}")
                
                # If we're still having issues, suggest OS-level process termination
                print("If process hangs, you may need to terminate it manually (Ctrl+C)")
        except asyncio.CancelledError as e:
            # Safely handle the cancellation error
            print(f"Caught cancellation during disconnect: {e}")
            raise Exception(f"Disconnect error: {e}")
        except Exception as e:
            # Handle any other exceptions during disconnect
            print(f"Error during disconnect: {e}")
            logger.error(f"Disconnect error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Disconnect error: {e}")
            
        # Ensure any remaining tasks are cleaned up
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, terminating...")
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc() 