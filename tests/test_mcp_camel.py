"""
Test script for MCP GitHub server integration with minimal dependencies.

This test:
1. Creates a minimal MCP client
2. Attempts to connect to the GitHub MCP server
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
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evoagentx.tools import MCPClient, MCPToolkit
from evoagentx.core.logging import logger

# Set a timeout for the entire test
TEST_TIMEOUT_SECONDS = 90  # Increased timeout

config = {
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-github"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "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"
            }
        }
    }
}

async def test_github_mcp():
    """Test connecting to GitHub MCP server and listing available tools."""
    logger.info("Starting GitHub MCP test...")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name
    
    try:
        # Create MCPToolkit with the config file
        toolkit = MCPToolkit(config_path=config_path)
        
        # Test connection
        logger.info("Connecting to MCP servers...")
        await toolkit.connect()
        
        # Verify connection
        assert toolkit.is_connected(), "Toolkit should be connected"
        assert len(toolkit.servers) == 1, "Should have one server configured"
        
        # Get available tools
        logger.info("Getting available tools...")
        tools = toolkit.get_tools()
        logger.info(f"Found {len(tools)} tools")
        
        # Verify tools
        assert len(tools) > 0, "Should have at least one tool available"
        
        # Test individual server
        server = toolkit.servers[0]
        assert server.is_connected(), "Server should be connected"
        assert server.command_or_url == "cmd", "Server command should match config"
        
        # Test tool listing
        available_tools = server.get_available_tools()
        logger.info(f"Available tools: {available_tools}")
        assert len(available_tools) > 0, "Server should have available tools"
        
        logger.info("GitHub MCP test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in GitHub MCP test: {e}")
        raise
    finally:
        # Cleanup
        try:
            await toolkit.disconnect()
            os.unlink(config_path)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

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
    # Check prerequisites
    if not is_npx_available():
        logger.error("npx is not available. Please install Node.js and npm first.")
        sys.exit(1)
    
    # Set up signal handling
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_and_exit(sig, frame, original_sigint))
    
    try:
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create timeout task
        timeout_task = loop.create_task(timeout_handler(TEST_TIMEOUT_SECONDS))
        
        # Run test
        test_task = loop.create_task(test_github_mcp())
        
        # Wait for either test completion or timeout
        done, pending = loop.run_until_complete(
            asyncio.wait([test_task, timeout_task], return_when=asyncio.FIRST_COMPLETED)
        )
        
        # Cancel timeout if test completed
        if not timeout_task.done():
            timeout_task.cancel()
        
        # Check test result
        if test_task in done:
            try:
                test_task.result()
                logger.info("Test completed successfully")
            except Exception as e:
                logger.error(f"Test failed: {e}")
                sys.exit(1)
        else:
            logger.error("Test did not complete before timeout")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error running test: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        loop.close()
        signal.signal(signal.SIGINT, original_sigint)

def cleanup_and_exit(sig, frame, original_handler):
    """Clean up resources and exit gracefully."""
    print("\nInterrupted. Cleaning up...")
    signal.signal(signal.SIGINT, original_handler)
    sys.exit(1)

if __name__ == "__main__":
    run_test()