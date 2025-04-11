"""
Unit tests for MCP tools integration.

These tests verify:
1. Loading MCP servers from config
2. Connecting to MCP servers
3. Getting available tools and their schemas
4. Proper error handling for invalid configurations
"""

import os
import json
import tempfile
import pytest
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch, Mock

from evoagentx.tools import MCPClient, MCPToolkit


# Sample MCP config for testing
SAMPLE_CONFIG = {
    "mcpServers": {
        "server1": {
            "url": "http://example.com/mcp",
            "timeout": 30,
            "headers": {
                "Authorization": "Bearer test_token"
            }
        },
        "server2": {
            "command": "npx",
            "args": ["-y", "mcprouter"],
            "env": {
                "SERVER_KEY": "test_key"
            }
        }
    }
}

# GitHub MCP Server Configuration
GITHUB_MCP_CONFIG = {
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


@pytest.fixture
def sample_config_file():
    """Create a temporary config file with sample MCP server definitions."""
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "mcp_config.json")
    
    with open(config_path, "w") as f:
        json.dump(SAMPLE_CONFIG, f, indent=2)
    
    yield config_path
    
    # Clean up
    try:
        os.remove(config_path)
        os.rmdir(temp_dir)
    except (FileNotFoundError, OSError):
        pass


@pytest.fixture
def github_config_file():
    """Create a temporary config file with GitHub MCP server definition."""
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "github_mcp_config.json")
    
    with open(config_path, "w") as f:
        json.dump(GITHUB_MCP_CONFIG, f, indent=2)
    
    yield config_path
    
    # Clean up
    try:
        os.remove(config_path)
        os.rmdir(temp_dir)
    except (FileNotFoundError, OSError):
        pass


@pytest.fixture
def mock_mcp_module():
    """Mock the MCP module for testing without actual connections."""
    # Mock ClientSession
    mock_session = AsyncMock()
    
    # Mock Tool object
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"}
        },
        "required": ["query"]
    }
    
    # Mock list_tools result
    mock_list_tools_result = MagicMock()
    mock_list_tools_result.tools = [mock_tool]
    mock_session.list_tools.return_value = mock_list_tools_result
    
    # Mock initialize
    mock_session.initialize.return_value = None
    
    # Mock streams
    mock_read_stream = AsyncMock()
    mock_write_stream = AsyncMock()
    
    # Mock context manager
    mock_stdio_context = MagicMock()
    mock_stdio_context.__aenter__ = AsyncMock(return_value=(mock_read_stream, mock_write_stream))
    mock_stdio_context.__aexit__ = AsyncMock()
    
    # Mock AsyncExitStack
    mock_exit_stack = AsyncMock()
    mock_exit_stack.enter_async_context = AsyncMock(return_value=(mock_read_stream, mock_write_stream))
    mock_exit_stack.aclose = AsyncMock()
    
    # Mock StdioServerParameters
    mock_stdio_params = MagicMock()
    
    with patch.multiple(
        "evoagentx.tools.mcp_tool",
        ClientSession=Mock(return_value=mock_session),
        StdioServerParameters=Mock(return_value=mock_stdio_params),
        stdio_client=Mock(return_value=mock_stdio_context),
        AsyncExitStack=Mock(return_value=mock_exit_stack)
    ):
        yield mock_session, mock_tool, mock_exit_stack, mock_stdio_context


class TestMCPClient:
    """Tests for the MCPClient class."""
    
    def test_init_url(self):
        """Test initializing with a URL."""
        client = MCPClient("https://example.com/mcp")
        assert client.command_or_url == "https://example.com/mcp"
        assert client._is_url == True
    
    def test_init_command(self):
        """Test initializing with a command."""
        client = MCPClient("npx", args=["-y", "mcprouter"])
        assert client.command_or_url == "npx"
        assert client._is_url == False
        assert client.args == ["-y", "mcprouter"]
    
    @pytest.mark.asyncio
    async def test_connect_command(self, mock_mcp_module):
        """Test connecting to a command-based server."""
        mock_session, mock_tool, mock_exit_stack, mock_stdio_context = mock_mcp_module
        
        client = MCPClient("npx", args=["-y", "mcprouter"])
        await client.connect()
        
        # Verify the StdioServerParameters was created with correct args
        from evoagentx.tools.mcp_tool import StdioServerParameters
        StdioServerParameters.assert_called_once_with(
            command="npx",
            args=["-y", "mcprouter"],
            env={**os.environ, **{}},  # Should merge with current environment
            cwd=os.getcwd()
        )
        
        # Verify stdio_client was called
        from evoagentx.tools.mcp_tool import stdio_client
        stdio_client.assert_called_once()
        
        # Verify AsyncExitStack was used
        assert mock_exit_stack.enter_async_context.called
        
        # Verify ClientSession was created with the streams
        from evoagentx.tools.mcp_tool import ClientSession
        ClientSession.assert_called_once()
        
        # Verify session is initialized
        assert client._session is not None
        mock_session.initialize.assert_called_once_with()
        assert mock_session.list_tools.called
        
        # Clean up
        await client.disconnect()
        
        # Verify exit stack was closed
        assert mock_exit_stack.aclose.called
    
    @pytest.mark.asyncio
    async def test_get_available_tools(self, mock_mcp_module):
        """Test getting available tools."""
        mock_session, mock_tool, mock_exit_stack, _ = mock_mcp_module
        
        client = MCPClient("npx", args=["-y", "mcprouter"])
        await client.connect()
        
        tools = client.get_available_tools()
        assert tools == [mock_tool.name]
        
        # Clean up
        await client.disconnect()


class TestMCPToolkit:
    """Tests for the MCPToolkit class."""
    
    def test_init_with_config(self, sample_config_file):
        """Test initializing with a config file."""
        toolkit = MCPToolkit(config_path=sample_config_file)
        assert len(toolkit.servers) == 2
        
        # Check server configurations
        server1 = toolkit.servers[0]
        assert server1.command_or_url == "http://example.com/mcp"
        assert server1.timeout == 30
        assert server1.headers == {"Authorization": "Bearer test_token"}
        
        server2 = toolkit.servers[1]
        assert server2.command_or_url == "npx"
        assert server2.args == ["-y", "mcprouter"]
        assert server2.env == {"SERVER_KEY": "test_key"}
    
    def test_init_with_invalid_config(self):
        """Test initializing with an invalid config path."""
        with pytest.raises(FileNotFoundError):
            MCPToolkit(config_path="nonexistent_config.json")
    
    def test_init_with_no_args(self):
        """Test initializing with no arguments."""
        with pytest.raises(ValueError):
            MCPToolkit()
    
    @pytest.mark.asyncio
    async def test_connect_and_get_tools(self, mock_mcp_module):
        """Test connecting and getting tools from all servers."""
        # This test only tests stdio connection, not HTTP since it's not implemented
        mock_session, mock_tool, mock_exit_stack, _ = mock_mcp_module
        
        # Create a mock server list (just one command-based server)
        server = MCPClient("npx", args=["-y", "mcprouter"])
        toolkit = MCPToolkit(servers=[server])
        
        # Connect
        await toolkit.connect()
        assert toolkit.is_connected() == True
        
        # Check the tools
        all_tools = toolkit.get_available_tools()
        assert "npx" in all_tools
        assert all_tools["npx"] == [mock_tool.name]
        
        # Get a flat list of all tools
        tool_names = toolkit.get_all_tool_names()
        assert len(tool_names) == 1
        assert mock_tool.name in tool_names
        
        # Clean up
        await toolkit.disconnect()
        assert toolkit.is_connected() == False


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


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("RUN_GITHUB_MCP_TEST") or not is_npx_available(), 
    reason="Requires actual GitHub MCP server and npx; set RUN_GITHUB_MCP_TEST=1 to run"
)
async def test_github_mcp_integration(github_config_file):
    """
    Integration test for GitHub MCP server.
    
    This test connects to the actual GitHub MCP server and requires:
    1. The 'npx' command to be available
    2. Internet connectivity
    3. The SERVER_KEY to be valid
    
    To run this test: RUN_GITHUB_MCP_TEST=1 pytest -xvs tests/unit/tools/test_mcp_tools.py::test_github_mcp_integration
    """
    # Create toolkit from config
    toolkit = MCPToolkit(config_path=github_config_file)
    
    try:
        # Connect to the server with a timeout
        try:
            await asyncio.wait_for(toolkit.connect(), timeout=20.0)
        except asyncio.TimeoutError:
            pytest.skip("Timeout while connecting to GitHub MCP server")
            return
        
        # Verify connection
        assert toolkit.is_connected() == True
        
        # Get available tools
        tools = toolkit.get_all_tool_names()
        assert len(tools) > 0
        
        # Check for expected GitHub tools
        github_tools = [
            "mcp_github_create_repository",
            "mcp_github_get_file_contents",
            "mcp_github_create_or_update_file"
        ]
        
        for tool in github_tools:
            if tool in tools:
                # Found at least one expected tool
                break
        else:
            pytest.fail("No expected GitHub tools found")
        
        # Verify schemas exist
        for server in toolkit.servers:
            for tool in server._mcp_tools:
                assert hasattr(tool, "name")
                assert hasattr(tool, "inputSchema")
                assert isinstance(tool.inputSchema, dict)
                assert "properties" in tool.inputSchema
    finally:
        # Clean up with a timeout
        try:
            await asyncio.wait_for(toolkit.disconnect(), timeout=5.0)
        except asyncio.TimeoutError:
            print("Timeout while disconnecting from GitHub MCP server")