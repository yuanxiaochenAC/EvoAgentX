#!/usr/bin/env python
"""
Example demonstrating how to use MCP dynamic functions with an agent in EvoAgentX.

This example shows how to:
1. Connect to an MCP server
2. Get dynamically generated functions from MCP tools
3. Integrate these functions with an agent to make them easy to call

To run this example:
```
python examples/mcp_agent_example.py
```

Note: You need to have an MCP server running or accessible before using this example.
"""

import asyncio
import json
import os
from typing import List, Callable, Dict, Any

from evoagentx.tools import MCPClient, MCPToolkit
from evoagentx.agents import Agent, AgentConfig  # Adjust import based on actual agent implementation
from evoagentx.core.logging import logger

# Replace with actual agent class from your project
# This is just a simple placeholder for demonstration purposes
class SimpleAgent:
    """Simple agent for demonstration purposes."""
    
    def __init__(self, name: str = "MCP Agent"):
        self.name = name
        self.tools: List[Callable] = []
        self.tool_names: List[str] = []
        
    def add_tool(self, tool: Callable):
        """Add a tool function to the agent."""
        self.tools.append(tool)
        self.tool_names.append(tool.__name__)
        
    def add_tools(self, tools: List[Callable]):
        """Add multiple tool functions to the agent."""
        for tool in tools:
            self.add_tool(tool)
    
    async def run_tool(self, tool_name: str, **kwargs):
        """Run a specific tool by name."""
        for tool in self.tools:
            if tool.__name__ == tool_name:
                return await tool(**kwargs)
        
        raise ValueError(f"Tool '{tool_name}' not found")
        
    def get_available_tools(self) -> List[str]:
        """Get names of all available tools."""
        return self.tool_names
        
    async def chat(self, message: str):
        """Chat with the agent."""
        response = f"I am {self.name} and I have these tools: {', '.join(self.tool_names)}"
        return response

# Example for integrating MCP functions with an agent
async def mcp_agent_example():
    logger.info("=== MCP Agent Integration Example ===")
    
    # Create an MCP client
    # Replace with your actual MCP server URL or command
    client = MCPClient(
        command_or_url="http://localhost:8000/mcp",
        timeout=30
    )
    
    # Create a simple agent
    agent = SimpleAgent(name="MCP-Enhanced Agent")
    
    try:
        # Connect to the server
        await client.connect()
        logger.info(f"Connected to MCP server: {client.is_connected()}")
        
        # Get tool functions from the MCP server
        tool_functions = client.get_tool_functions()
        logger.info(f"Found {len(tool_functions)} MCP tool functions")
        
        # Add the MCP tool functions to the agent
        agent.add_tools(tool_functions)
        logger.info(f"Added MCP tools to agent, available tools: {agent.get_available_tools()}")
        
        # Demonstrate calling a tool through the agent (if any available)
        available_tools = agent.get_available_tools()
        if available_tools:
            # Get the first available tool for demonstration
            tool_name = available_tools[0]
            logger.info(f"Calling tool through agent: {tool_name}")
            
            # This is just a placeholder - you'll need to know the proper parameters
            # for the specific tool you're calling
            try:
                # Note: In a real implementation, you would know what parameters 
                # the specific tool requires
                result = await agent.run_tool(tool_name)
                logger.info(f"Result: {result}")
            except Exception as e:
                logger.warning(f"Couldn't call tool {tool_name}: {e}")
                logger.info("This is expected if the tool requires parameters.")
        
        # Chat with the agent to show integration
        response = await agent.chat("What tools do you have?")
        logger.info(f"Agent response: {response}")
        
    except Exception as e:
        logger.error(f"Error in MCP agent example: {e}")
    finally:
        # Disconnect from the server
        await client.disconnect()
        logger.info("Disconnected from MCP server")

# Example for using multiple MCP servers with an agent
async def multi_server_agent_example():
    logger.info("\n=== Multi-Server MCP Agent Example ===")
    
    # Create MCP clients for different servers
    # Replace with your actual MCP server URLs or commands
    client1 = MCPClient(
        command_or_url="http://localhost:8000/mcp",
        timeout=30
    )
    
    client2 = MCPClient(
        command_or_url="http://localhost:8001/mcp",
        timeout=30
    )
    
    # Create a toolkit to manage multiple servers
    toolkit = MCPToolkit(servers=[client1, client2])
    
    # Create a simple agent
    agent = SimpleAgent(name="Multi-MCP Agent")
    
    try:
        # Connect to all servers
        await toolkit.connect()
        logger.info(f"Connected to MCP servers: {toolkit.is_connected()}")
        
        # Get tool functions from all MCP servers
        tool_functions = toolkit.get_all_tool_functions()
        logger.info(f"Found {len(tool_functions)} total MCP tool functions")
        
        # Add all MCP tool functions to the agent
        agent.add_tools(tool_functions)
        logger.info(f"Added MCP tools to agent, available tools: {agent.get_available_tools()}")
        
        # Chat with the agent to show integration
        response = await agent.chat("What tools do you have?")
        logger.info(f"Agent response: {response}")
        
    except Exception as e:
        logger.error(f"Error in multi-server MCP agent example: {e}")
    finally:
        # Disconnect from all servers
        await toolkit.disconnect()
        logger.info("Disconnected from all MCP servers")

# Detailed example showing how to create a real EvoAgentX agent with MCP tools
async def evoagentx_agent_example():
    logger.info("\n=== EvoAgentX Agent with MCP Tools Example ===")
    
    # Skip this example if the agent hasn't been imported successfully
    if 'Agent' not in globals() or 'AgentConfig' not in globals():
        logger.warning("Skipping EvoAgentX agent example due to missing imports")
        return
    
    # Create an MCP client
    client = MCPClient(
        command_or_url="http://localhost:8000/mcp",
        timeout=30
    )
    
    try:
        # Connect to the server
        await client.connect()
        logger.info(f"Connected to MCP server: {client.is_connected()}")
        
        # Get tool functions from the MCP server
        tool_functions = client.get_tool_functions()
        logger.info(f"Found {len(tool_functions)} MCP tool functions")
        
        # Create an actual EvoAgentX agent with the MCP tools
        agent_config = AgentConfig(
            name="MCP-Enhanced Agent",
            description="An agent with MCP tool capabilities",
            # Add other configuration as needed
        )
        
        # Create the agent with the MCP tool functions
        agent = Agent(config=agent_config)
        
        # Add MCP tools to the agent
        for tool_func in tool_functions:
            agent.add_tool(tool_func)
            
        logger.info(f"Created EvoAgentX agent with {len(tool_functions)} MCP tools")
        
        # Example of using the agent - update with actual usage pattern for your Agent class
        # This is a placeholder and should be replaced with actual EvoAgentX Agent usage
        # response = await agent.chat("What tools do you have available?")
        # logger.info(f"Agent response: {response}")
        
    except Exception as e:
        logger.error(f"Error in EvoAgentX agent example: {e}")
    finally:
        # Disconnect from the server
        await client.disconnect()
        logger.info("Disconnected from MCP server")

async def main():
    # Run the examples
    await mcp_agent_example()
    await multi_server_agent_example()
    
    # Run the EvoAgentX-specific example if the imports worked
    # This is in a try-except to handle the case where the imports might fail
    try:
        await evoagentx_agent_example()
    except NameError:
        # If the imports weren't done properly, we skip this example
        logger.warning("Skipping EvoAgentX agent example due to missing imports")

if __name__ == "__main__":
    asyncio.run(main()) 