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
"""

import asyncio
import json
from evoagentx.tools import MCPClient, MCPToolkit
from evoagentx.core.logging import logger

# Sample config for demonstration
SAMPLE_CONFIG = {
    "mcpServers": {
        "server1": {
            "url": "http://localhost:8000/mcp",
            "timeout": 30
        }
    }
}

# Save sample config to a file
def create_sample_config():
    with open("mcp_config.json", "w") as f:
        json.dump(SAMPLE_CONFIG, f, indent=4)
    logger.info("Created sample config file: mcp_config.json")

# Example for single MCP server
async def mcp_client_example():
    logger.info("=== MCPClient Example ===")
    
    # Create an MCP client
    # Replace with your actual MCP server URL or command
    client = MCPClient(
        command_or_url="http://localhost:8000/mcp",
        timeout=30
    )
    
    try:
        # Connect to the server
        await client.connect()
        logger.info(f"Connected: {client.is_connected()}")
        
        # List available tools
        available_tools = client.get_available_tools()
        logger.info(f"Available tools: {available_tools}")
        
        # Show tool info
        tool_info = client.get_tool_info()
        logger.info(f"Tool info: {json.dumps(tool_info, indent=2)}")
        
        # If tools are available, you can call them
        if available_tools:
            tool_name = available_tools[0]
            logger.info(f"Calling tool: {tool_name}")
            
            # This is just a placeholder - you'll need to provide the actual
            # parameters required by the tool you're calling
            # result = await client._session.call_tool(tool_name, {})
            # logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Error in MCPClient example: {e}")
    finally:
        # Disconnect
        await client.disconnect()
        logger.info("Disconnected")

# Example for multiple MCP servers
async def mcp_toolkit_example():
    logger.info("\n=== MCPToolkit Example ===")
    
    try:
        # Create sample config for demonstration
        create_sample_config()
        
        # Create a toolkit from config file
        toolkit = MCPToolkit(config_path="mcp_config.json")
        
        # Connect to all servers
        async with toolkit.connection() as connected_toolkit:
            logger.info(f"Connected: {connected_toolkit.is_connected()}")
            
            # Get available tools from all servers
            tools_by_server = connected_toolkit.get_available_tools()
            logger.info(f"Tools by server: {json.dumps(tools_by_server, indent=2)}")
            
            # Get a flat list of all tools
            all_tools = connected_toolkit.get_all_tool_names()
            logger.info(f"All tools: {all_tools}")
            
            # Show toolkit info
            toolkit_info = connected_toolkit.get_tool_info()
            logger.info(f"Toolkit info: {json.dumps(toolkit_info, indent=2)}")
            
        # Toolkit automatically disconnects when the context manager exits
        logger.info("Disconnected from all servers")
    except Exception as e:
        logger.error(f"Error in MCPToolkit example: {e}")

async def main():
    # Run the examples
    await mcp_client_example()
    await mcp_toolkit_example()

if __name__ == "__main__":
    asyncio.run(main()) 