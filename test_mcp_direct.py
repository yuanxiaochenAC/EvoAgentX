#!/usr/bin/env python
"""
Minimal test for MCP GitHub integration using direct SDK.

This script is a minimal test that directly uses the MCP SDK without our custom
abstraction layer, to help isolate the issue with GitHub MCP server connections.
"""

import os
import asyncio
import sys
import signal
from contextlib import AsyncExitStack

try:
    import mcp
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("MCP Python SDK not found. Please install it with 'pip install mcp'")
    sys.exit(1)

# Test timeout
TEST_TIMEOUT_SECONDS = 90

async def test_github_mcp():
    """Test connecting to GitHub MCP server directly using the SDK."""
    print("=== Minimal MCP GitHub Integration Test ===")
    
    # Start timer to prevent hanging
    timer_task = asyncio.create_task(timeout_handler(TEST_TIMEOUT_SECONDS))
    exit_stack = None
    session = None
    
    try:
        print("Connecting to GitHub MCP server...")
        
        # Setup server parameters
        server_params = StdioServerParameters(
            command="npx",
            args=["--yes", "@modelcontextprotocol/server-github"],
            env={
                **os.environ,
                "GITHUB_PERSONAL_ACCESS_TOKEN": "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"
            },
            cwd=os.getcwd()
        )
        
        # Create resource management stack
        exit_stack = AsyncExitStack()
        
        # Connect to the server
        streams = await exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = streams
        
        # Create the MCP session
        session = ClientSession(read_stream, write_stream)
        
        print("Server connected, initializing MCP session...")
        
        # Initialize the session with a longer timeout
        await asyncio.wait_for(session.initialize(), timeout=60.0)
        
        print("MCP session initialized, fetching available tools...")
        
        # Get available tools
        result = await asyncio.wait_for(session.list_tools(), timeout=30.0)
        tools = result.tools
        
        print(f"Successfully connected to GitHub MCP server")
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__} - {str(e)}")
    finally:
        # Clean up resources
        if timer_task and not timer_task.done():
            timer_task.cancel()
            try:
                await timer_task
            except asyncio.CancelledError:
                pass
        
        # Clean up MCP resources
        if session:
            print("Closing MCP session...")
        
        if exit_stack:
            print("Cleaning up resources...")
            await exit_stack.aclose()
            print("Resources cleaned up")

async def timeout_handler(seconds):
    """Handle timeout by exiting gracefully after specified seconds."""
    try:
        await asyncio.sleep(seconds)
        print(f"Test timed out after {seconds} seconds")
        os._exit(1)  # Force exit
    except asyncio.CancelledError:
        pass  # Task was cancelled, test completed

def run_test():
    """Run the test asynchronously."""
    try:
        # Set up signal handler for Ctrl+C
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda sig, frame: cleanup_and_exit(sig, frame, original_sigint))
        
        # Run the test
        asyncio.run(test_github_mcp())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Error running test: {e}")

def cleanup_and_exit(sig, frame, original_handler):
    """Clean up resources and exit gracefully."""
    print("\nInterrupted. Cleaning up...")
    signal.signal(signal.SIGINT, original_handler)
    sys.exit(1)

if __name__ == "__main__":
    run_test() 