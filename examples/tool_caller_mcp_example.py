#!/usr/bin/env python
"""
Example demonstrating how to use the ToolCaller agent with MCP tools in EvoAgentX.

This example shows how to:
1. Connect to an MCP server
2. Get tool descriptions from the MCP server
3. Use the ToolCaller agent to decide whether to call MCP tools or provide direct answers

To run this example:
```
python examples/tool_caller_mcp_example.py
```

Note: You need to have an MCP server running or accessible before using this example.
"""

import asyncio
import os
import json
from typing import Dict, Any, List

from evoagentx.agents.tool_caller import ToolCaller
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.tools.mcp import MCPClient
from evoagentx.core.logging import logger
from evoagentx.tools.mcp import MCPToolkit
from evoagentx.core.message import MessageType


async def main():
    logger.info("=== ToolCaller Agent with MCP Tools Example ===")
    
    # Get OpenAI API key
    api_key = "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"
    
    # Initialize the language model
    llm_config = OpenAILLMConfig(
        llm_type="OpenAILLM",
        model="gpt-4o-mini",  # Using GPT-4o-mini for tool usage capabilities
        openai_key=api_key,
        temperature=0.7,
        max_tokens=1000,
    )
    
    # Create MCP client and toolkit
    config_path = "tests/mcp.config"
    toolkit = MCPToolkit(config_path=config_path)
    
    try:
        # Connect to the MCP server
        await toolkit.connect()
        logger.info(f"Connected to MCP server: {toolkit.is_connected()}")
        
        # Create ToolCaller agent
        agent = ToolCaller(llm_config=llm_config)
        
        # Get tools from the MCP toolkit
        mcp_tools = toolkit.get_tools()
        
        # Add tools to the agent using the new method
        agent.add_tools_from_toolkit(mcp_tools)
        
        logger.info(f"Found {len(agent.tool_descriptions)} MCP tools")
        
        # Example queries - adjust based on your MCP server's available tools
        queries = [
            "What is 2 + 2?",  # likely direct answer
            # "What tools do you have available?",  # likely direct answer
            # Add queries specific to your MCP tools
            # For example, if you have a weather tool:
            # "What's the weather in London?"
        ]
        
        # Process each query
        for query in queries:
            logger.info(f"\nProcessing query: {query}")
            
            # Process the query using execute
            try:
                result = await agent.execute(
                    action_name=agent.tool_caller_action_name,
                    action_input_data={"query": query},
                    return_msg_type=MessageType.RESPONSE
                )
                
                # Check if it's a direct answer or tool call
                if result.action == "direct_answer":
                    logger.info(f"Direct answer: {result.content}")
                else:
                    # Get the tool result from the message content
                    tool_output = result.content.result
                    logger.info(f"Tool call result: {json.dumps(tool_output, indent=2) if isinstance(tool_output, dict) else str(tool_output)}")
            except Exception as e:
                logger.error(f"Error processing query: {e}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up resources
        await toolkit.disconnect()
        logger.info("Disconnected from MCP server")
    
    logger.info("\nExample completed!")

if __name__ == "__main__":
    asyncio.run(main())