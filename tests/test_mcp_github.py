"""
Test script for MCP GitHub server integration with minimal dependencies.

This test:
1. Starts the GitHub MCP server as a separate process
2. Connects to it via HTTP/SSE
3. Gets the list of available tools

Note: This is a standalone script that can be run directly:
    python tests/test_mcp_github.py
"""

import os
import json
import asyncio
import tempfile
import sys
import signal
import subprocess
import time
from pathlib import Path

# Add the project root to the Python path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evoagentx.tools import MCPToolkit
from evoagentx.core.logging import logger

# Set a timeout for the entire test
TEST_TIMEOUT_SECONDS = 90  # Increased timeout
MCP_SERVER_PORT = 3000  # Port for the MCP server

# GitHub MCP Server Configuration
config = {
    "mcpServers": {
        "github": {
            "url": f"http://localhost:{MCP_SERVER_PORT}",
            "timeout": 30
        }
    }
}

async def start_mcp_server():
    """Start the GitHub MCP server as a separate process."""
    logger.info("Starting GitHub MCP server...")
    
    # Start the server with npx
    process = subprocess.Popen(
        ["npx", "-y", "@modelcontextprotocol/server-github", "--port", str(MCP_SERVER_PORT)],
        env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": "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"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    await asyncio.sleep(5)  # Give the server time to start
    
    return process

async def test_github_mcp():
    """Test GitHub MCP server integration."""
    logger.info("=== GitHub MCP Server Integration Test (Minimal) ===")
    
    # Setup timer to prevent hanging
    timer_task = None
    toolkit = None
    server_process = None
    
    try:
        # Check if npx is available (required for GitHub MCP)
        if not is_npx_available():
            logger.error("The 'npx' command is not available in your environment. "
                         "Please install Node.js and npm to use the GitHub MCP server.")
            return
        
        # Start timer to prevent hanging
        timer_task = asyncio.create_task(timeout_handler(TEST_TIMEOUT_SECONDS))
        
        # Start the MCP server
        server_process = await start_mcp_server()
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            config_path = f.name
        
        try:
            # Create MCPToolkit with the config file
            toolkit = MCPToolkit(config_path=config_path)
            
            # Test connection using context manager
            logger.info("Connecting to MCP servers...")
            async with toolkit.connection() as connected_toolkit:
                # Verify connection
                assert connected_toolkit.is_connected(), "Toolkit should be connected"
                assert len(connected_toolkit.servers) == 1, "Should have one server configured"
                
                # Get available tools
                logger.info("Getting available tools...")
                tools = connected_toolkit.get_available_tools()
                logger.info(f"Found tools: {json.dumps(tools, indent=2)}")
                
                # Verify tools
                assert len(tools) > 0, "Should have at least one tool available"
                
                # Test individual server
                server = connected_toolkit.servers[0]
                assert server.is_connected(), "Server should be connected"
                assert server.command_or_url == f"http://localhost:{MCP_SERVER_PORT}", "Server URL should match config"
                
                # Test tool listing
                available_tools = server.get_available_tools()
                logger.info(f"Available tools: {available_tools}")
                assert len(available_tools) > 0, "Server should have available tools"
                
                logger.info("GitHub MCP test completed successfully")
                
        finally:
            # Cleanup config file
            try:
                os.unlink(config_path)
            except Exception as e:
                logger.error(f"Error removing config file: {e}")
            
            # Stop the MCP server
            if server_process:
                logger.info("Stopping MCP server...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cancel timeout handler
        if timer_task and not timer_task.done():
            timer_task.cancel()
            try:
                await timer_task
            except asyncio.CancelledError:
                pass

async def timeout_handler(seconds):
    """Handle timeout by exiting gracefully after specified seconds."""
    try:
        await asyncio.sleep(seconds)
        logger.error(f"Test timed out after {seconds} seconds")
        # Force exit the process to prevent hanging
        os._exit(1)
    except asyncio.CancelledError:
        # Task was cancelled, meaning test completed before timeout
        pass

def is_npx_available():
    """Check if npx is available in the system."""
    import subprocess
    try:
        subprocess.run(["npx", "--version"], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       check=False)
        return True
    except FileNotFoundError:
        return False

def run_test():
    """Run the GitHub MCP test asynchronously."""
    try:
        # Set a signal handler to exit gracefully on Ctrl+C
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda sig, frame: cleanup_and_exit(sig, frame, original_sigint))
        
        # Run the test with asyncio
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