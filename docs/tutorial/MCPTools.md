# EvoAgentX MCP Tools and ToolCaller Tutorial

This document provides a step-by-step guide to using the Model Context Protocol (MCP) tools and the ToolCaller agent in EvoAgentX. These powerful components allow you to connect to external services and execute complex tasks through a standardized interface.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
  - [MCP Config File](#mcp-config-file)
  - [Environment Variables](#environment-variables)
- [MCP Toolkit](#mcp-toolkit)
  - [Connecting to MCP Servers](#connecting-to-mcp-servers)
  - [Listing Available Tools](#listing-available-tools)
- [ToolCaller Agent](#toolcaller-agent)
  - [Initializing the Agent](#initializing-the-agent)
  - [Adding MCP Tools](#adding-mcp-tools)
  - [Understanding ToolCaller Actions](#understanding-toolcaller-actions)
  - [Executing ToolCalling Action](#executing-toolcalling-action)
  - [Executing ToolCallSummarizing Action](#executing-toolcallsummarizing-action)
- [Complete Example](#complete-example)
  - [Project Analysis Example](#project-analysis-example)
- [Best Practices](#best-practices)
  - [Error Handling](#error-handling)
  - [Graceful Shutdown](#graceful-shutdown)

## Overview

The Model Context Protocol (MCP) is a standardized way for language models to interact with external tools and services. EvoAgentX provides robust support for MCP through:

1. **MCPClient**: A client for connecting to individual MCP servers
2. **MCPToolkit**: A manager for multiple MCP servers and their tools
3. **ToolCaller**: An agent that can use MCP tools to complete tasks

These components together enable AI agents to access external capabilities like file systems, GitHub repositories, PDF analysis, and more.

## Configuration

### MCP Config File

MCP servers are configured through a JSON configuration file. Each server entry specifies:

- The command to start the server
- Command-line arguments
- Environment variables needed for authentication

Here's an example configuration file (`mcp.config`):

```json
{
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-github"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "your_github_pat_here"
            }
        },
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/path/to/your/project/"
            ]
        },
        "pdf-reader": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-pdf-reader"
            ],
            "name": "PDF Reader"
        }
    }
}
```

Each server configuration can include:

- `command`: The executable to run
- `args`: Command-line arguments
- `env`: Environment variables needed by the server
- `name` (optional): A custom name for the server

### Environment Variables

In addition to server-specific configuration, you'll need to set up environment variables for your LLM:

```
# OpenAI API Key for the ToolCaller agent
OPENAI_API_KEY=your_openai_api_key_here
```

You can set these in a `.env` file or directly in your environment.

## MCP Toolkit

### Connecting to MCP Servers

The `MCPToolkit` class manages multiple MCP servers from your configuration file:

```python
from evoagentx.tools import MCPToolkit

# Initialize the toolkit with your config
toolkit = MCPToolkit(config_path="path/to/mcp.config")

# Connect to all configured servers
await toolkit.connect()

# Check connection status
if toolkit.is_connected():
    print("Successfully connected to MCP servers!")
```

### Listing Available Tools

After connecting, you can retrieve the available tools in OpenAI-compatible format:

```python
# Get all available tools as OpenAI tool schemas
tool_schemas = toolkit.get_all_openai_tool_schemas()
print(f"Available tools: {json.dumps(tool_schemas, indent=2)}")
```

## ToolCaller Agent

### Initializing the Agent

The `ToolCaller` agent integrates with language models to execute tools based on natural language inputs:

```python
from evoagentx.agents.tool_caller import ToolCaller
from evoagentx.models.model_configs import OpenAILLMConfig

# Configure the language model
llm_config = OpenAILLMConfig(
    llm_type="OpenAILLM",
    model="gpt-4o-mini",  # Any model with tool-calling capabilities
    openai_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)

# Initialize the ToolCaller agent
# max_tool_try parameter controls the maximum recursive depth of tool calls
tool_caller_agent = ToolCaller(llm_config=llm_config, max_tool_try=2)
```

### Adding MCP Tools

You can add your MCP toolkit to the ToolCaller agent:

```python
# Add the MCP toolkit to the agent
tool_caller_agent.add_mcp_toolkit(toolkit)
```

### Understanding ToolCaller Actions

The ToolCaller agent provides two main actions:

1. **ToolCalling**: This action processes a user query, decides which tools to call, executes them, and returns the results. It can make multiple sequential tool calls up to the `max_tool_try` limit.

2. **ToolCallSummarizing**: This action takes the results from tool calls and generates a concise, human-readable summary of the findings.

These actions are pre-configured in the ToolCaller agent and can be accessed as follows:

```python
# Access the action names
tool_calling_action = tool_caller_agent.tool_calling_action_name       # "tool_calling" by default
summarizing_action = tool_caller_agent.tool_summarizing_action_name    # "tool_call_summarizing" by default
```

### Executing ToolCalling Action

The ToolCalling action processes a user query, determines which tools to use, and executes them:

```python
from evoagentx.core.message import Message, MessageType

# Create a user message with instructions
user_query = "Please analyze the repository 'openai/openai-python' and summarize its main features."
input_message = Message(content=user_query, msg_type=MessageType.REQUEST, agent="user")

# Execute the tool calling action
tool_result = await tool_caller_agent.execute(
    action_name=tool_caller_agent.tool_calling_action_name,
    msgs=input_message,
    return_msg_type=MessageType.RESPONSE,
)

print(f"Tool execution result: {tool_result}")
```

The ToolCalling action:
- Analyzes the user query
- Determines which tools are needed
- Executes those tools sequentially
- May make recursive tool calls (up to max_tool_try limit)
- Returns the complete tool execution results

### Executing ToolCallSummarizing Action

After executing tools, you can generate a concise summary of the results:

```python
from evoagentx.actions.tool_calling import SUMMARIZING_PROMPT

# Create a summary of the tool execution results
summary_result = await tool_caller_agent.execute(
    action_name=tool_caller_agent.tool_summarizing_action_name,
    msgs=input_message,                     # Original user query
    history=[tool_result],                  # Results from the tool calling
    sys_msg=SUMMARIZING_PROMPT,             # Optional custom system prompt
    return_msg_type=MessageType.RESPONSE
)

print(f"Summary: {summary_result}")
```

The ToolCallSummarizing action:
- Takes the original query and tool execution results
- Analyzes the information to extract key findings
- Generates a concise, user-friendly summary
- Returns a formatted response based on the system prompt

You can customize the summarization behavior with a custom system prompt:

```python
# Import the default summarizing prompt
from evoagentx.actions.tool_calling import SUMMARIZING_PROMPT

# Customize it or create your own
custom_prompt = """
You are an expert summarizer. Create a concise summary of the tool execution results,
focusing on the following aspects:
1. Key findings
2. Actionable insights
3. Any limitations or issues encountered

Your summary should be clear, informative, and directly address the user's original query.
"""

# Use the custom prompt
summary_result = await tool_caller_agent.execute(
    action_name=tool_caller_agent.tool_summarizing_action_name,
    msgs=input_message,
    history=[tool_result],
    sys_msg=custom_prompt,
    return_msg_type=MessageType.RESPONSE
)
```

## Complete Example

Here's a complete example that demonstrates:
1. Connecting to MCP servers
2. Initializing a ToolCaller agent
3. Analyzing a GitHub project
4. Summarizing the results

### Project Analysis Example

```python
import asyncio
import os
import signal
import traceback
import dotenv
from evoagentx.tools import MCPToolkit
from evoagentx.core.logging import logger
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.agents.tool_caller import ToolCaller
from evoagentx.core.message import Message, MessageType
from evoagentx.actions.tool_calling import SUMMARIZING_PROMPT

# Load environment variables
dotenv.load_dotenv()

# Set up signal handling for graceful shutdown
def handle_exit_signal(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    os._exit(0)

signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

async def main():
    logger.info("=== ToolCaller Agent with MCP Tools Example ===")
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Initialize the language model
    llm_config = OpenAILLMConfig(
        llm_type="OpenAILLM",
        model="gpt-4o-mini",
        openai_key=api_key,
        temperature=0.7,
    )
    
    # Initialize the MCP toolkit
    config_path = "examples/mcp.config"
    toolkit = MCPToolkit(config_path=config_path)
    
    try:
        # Connect to MCP servers
        await toolkit.connect()
        logger.info(f"Connected to MCP servers: {toolkit.is_connected()}")
        
        # Create ToolCaller agent
        tool_caller_agent = ToolCaller(llm_config=llm_config, max_tool_try=2)
        tool_caller_agent.add_mcp_toolkit(toolkit)
        
        # Define the project to analyze
        project_name = "camel-ai/owl"
        
        # Create a user query
        user_query = f"""
        Assume you are a tech lead looking for open projects that might be useful for your task.
        We are currently looking into {project_name} project. 
        Please look into this project and provide a comprehensive summary.
        """
        
        # Create a message object
        input_message = Message(content=user_query, msg_type=MessageType.REQUEST, agent="user")
        
        # Step 1: Execute tool calling to gather information
        logger.info("Executing tool calls to gather information...")
        tool_call_result = await tool_caller_agent.execute(
            action_name=tool_caller_agent.tool_calling_action_name,
            msgs=input_message,
            return_msg_type=MessageType.RESPONSE,
        )
        
        # Step 2: Generate a summary of the gathered information
        logger.info("Generating summary of tool call results...")
        summary_result = await tool_caller_agent.execute(
            action_name=tool_caller_agent.tool_summarizing_action_name,
            msgs=input_message,
            history=[tool_call_result],
            sys_msg=SUMMARIZING_PROMPT,
            return_msg_type=MessageType.RESPONSE
        )
        
        # Print the final summary
        print(f"Summarizing output: {summary_result}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        # Raise the exception with a full traceback for better debugging
        raise Exception(f"Traceback: {traceback.format_exc()}")
    finally:
        # Clean up resources
        print("Disconnecting from MCP servers")
        try:
            await toolkit.disconnect()
        except asyncio.CancelledError as e:
            # Safely handle the cancellation error
            print(f"Caught cancellation during disconnect: {e}")
            print("This is expected behavior with the MCP client and can be safely ignored.")
        except Exception as e:
            print(f"Error during disconnect: {e}")
        logger.info("Disconnected from MCP servers")
    
    logger.info("\nExample completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, terminating...")
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

## Best Practices

### Error Handling

When working with MCP tools, implement robust error handling:

1. **Connection Errors**: Handle connection failures gracefully
2. **Tool Execution Errors**: Capture and process exceptions from tool calls
3. **Timeout Handling**: Set timeouts for operations that might hang

```python
try:
    # Attempt to perform a tool call
    result = await tool_caller_agent.execute(...)
except Exception as e:
    logger.error(f"Tool execution failed: {e}")
    # Implement fallback strategy
```

### Graceful Shutdown

Ensure proper cleanup of resources, especially for background MCP servers:

```python
try:
    # Main application logic
    await toolkit.connect()
    # ... perform operations ...
finally:
    # Always disconnect, even if exceptions occur
    try:
        await toolkit.disconnect()
    except asyncio.CancelledError as e:
        # Safely handle the cancellation error
        print(f"This is expected behavior with the MCP client")
    except Exception as e:
        logger.error(f"Error during disconnect: {e}")
```

For long-running applications, implement signal handlers to gracefully shut down when receiving termination signals:

```python
def setup_signal_handlers():
    def handle_exit(signum, frame):
        print(f"Received signal {signum}, shutting down...")
        # Clean up resources
        asyncio.create_task(toolkit.disconnect())
        # Exit after a brief delay
        loop = asyncio.get_event_loop()
        loop.call_later(2, loop.stop)
    
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
```

By following this tutorial, you should now be able to effectively use MCP tools and the ToolCaller agent in your EvoAgentX applications to extend your AI agents with external capabilities. 