#!/usr/bin/env python
"""
Simple test for MCP GitHub integration based on official examples.

This script follows the patterns from the official MCP Python SDK documentation
to test connecting to and using the GitHub MCP server.
"""

import os
import asyncio
import sys
import json
from contextlib import AsyncExitStack

try:
    from mcp import ClientSession, types
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:
    print("MCP Python SDK not found. Please install it with 'pip install mcp'")
    sys.exit(1)

# GitHub Personal Access Token (example token - replace with your own)
GITHUB_PAT = "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"

# Optional: Define a sampling callback for model responses
async def handle_sampling_message(
    message: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Example response from model",
        ),
        model="gpt-3.5-turbo",
        stopReason="endTurn",
    )

async def test_github_mcp():
    """Test connecting to GitHub MCP server using the SDK patterns."""
    print("=== Testing GitHub MCP Server Integration ===")
    
    # Create server parameters for the GitHub MCP server
    server_params = StdioServerParameters(
        command="npx",  # Executable
        args=["--yes", "@modelcontextprotocol/server-github"],  # Command line arguments
        env={
            **os.environ,
            "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_PAT,
        },
        cwd=os.getcwd(),
    )
    
    print("Connecting to GitHub MCP server...")
    
    # Use the stdio_client pattern from the examples
    try:
        async with stdio_client(server_params) as (read, write):
            print("Connection established, creating session...")
            
            # Create a session with the sampling callback
            session = ClientSession(
                read,
                write,
                sampling_callback=handle_sampling_message
            )
            
            # Initialize the connection with a much longer timeout (120 seconds)
            # Since we observed the server is starting but taking time to initialize
            print("Initializing MCP session (this may take several minutes)...")
            try:
                print("Waiting for initialization... (timeout: 120s)")
                await asyncio.wait_for(session.initialize(), timeout=120.0)
                print("Session initialized successfully!")
                
                # List available tools
                print("Listing available tools...")
                tools_result = await asyncio.wait_for(session.list_tools(), timeout=30.0)
                tools = tools_result.tools
                
                print(f"Found {len(tools)} available tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description or 'No description'}")
                    print(f"    Schema: {json.dumps(tool.inputSchema, indent=2)}")
                
            except asyncio.TimeoutError:
                print("ERROR: Timeout during MCP session initialization")
                print("The GitHub MCP server started but did not initialize within the timeout period.")
                print("This could be due to:")
                print("- The server is still setting up its connections to GitHub")
                print("- The token might be invalid or missing required scopes")
                print("- There might be network connectivity issues with GitHub API")
            except Exception as e:
                print(f"ERROR: {type(e).__name__} - {str(e)}")
            finally:
                print("Closing MCP session...")
    except Exception as e:
        print(f"ERROR connecting to MCP server: {type(e).__name__} - {str(e)}")

def run_test():
    """Run the GitHub MCP test."""
    try:
        # Set a custom event loop policy if needed for Windows
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run the test
        asyncio.run(test_github_mcp())
        print("Test completed.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"ERROR running test: {type(e).__name__} - {str(e)}")

if __name__ == "__main__":
    run_test() 