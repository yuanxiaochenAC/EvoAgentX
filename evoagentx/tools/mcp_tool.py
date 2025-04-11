"""
MCP (Model Context Protocol) Tool for EvoAgentX.

This tool allows integration with various MCP servers, enabling tools and capabilities
from remote services to be used within EvoAgentX.

Based on the MCP Python SDK (https://github.com/modelcontextprotocol/python-sdk).
"""

import os
import json
import inspect
import sys
import asyncio
import signal
import aiohttp
from contextlib import asynccontextmanager, AsyncExitStack
from typing import Any, List, Dict, Optional, AsyncGenerator, Union, Callable, cast, TextIO, Tuple
from urllib.parse import urlparse

from .tool import Tool
from ..core.logging import logger

try:
    import mcp
    from mcp import ClientSession, ListToolsResult, Tool as MCPTool
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.sse import sse_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    logger.error("MCP Python SDK not found. Please install it with 'pip install mcp'")


class MCPClient(Tool):
    """
    Client for connecting to and interacting with MCP servers.
    
    This class provides an abstraction layer to interact with Model Context
    Protocol (MCP) servers, either as local processes or remote endpoints.
    
    Args:
        command_or_url (str): Command to launch a local MCP server or a URL
            pointing to a remote MCP server.
        args (List[str], optional): Command-line arguments for a local MCP
            server. Ignored if connecting to a remote server. Defaults to [].
        env (Dict[str, str], optional): Environment variables for a local MCP
            server. Ignored if connecting to a remote server. Defaults to {}.
        timeout (Optional[int], optional): Connection timeout in seconds.
            Defaults to None.
        headers (Optional[Dict[str, str]], optional): HTTP headers to send
            when connecting to a remote server. Ignored for local servers.
            Defaults to None.
    """
    
    def __init__(
        self,
        command_or_url: str,
        args: List[str] = None,
        env: Dict[str, str] = None,
        timeout: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the MCP client."""
        super().__init__()
        
        if not HAS_MCP:
            raise ImportError("MCP Python SDK not found. Please install it with 'pip install mcp'")
            
        self.command_or_url = command_or_url
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.headers = headers or {}
        
        # Will be initialized when connecting
        self._session: Optional[ClientSession] = None
        self._mcp_tools: List["MCPTool"] = []
        self._is_url = self._check_if_url(command_or_url)
        self._exit_stack = AsyncExitStack()
        self._is_connected = False
    
    def _check_if_url(self, command_or_url: str) -> bool:
        """Check if the given string is a URL."""
        return command_or_url.startswith(("http://", "https://"))
    
    async def _wait_for_server(self, url: str, max_retries: int = 5, retry_delay: float = 1.0) -> bool:
        """Wait for the server to be ready by checking the connection."""
        timeout = aiohttp.ClientTimeout(total=5)  # 5 second timeout for each attempt
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return True
            except asyncio.TimeoutError:
                logger.debug(f"Server connection timed out (attempt {attempt + 1}/{max_retries})")
            except aiohttp.ClientError as e:
                logger.debug(f"Server connection error (attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                logger.debug(f"Unexpected error checking server (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                return False
        return False
    
    async def connect(self):
        """
        Connect to the MCP server.
        
        Returns:
            MCPClient: The connected client instance.
        """
        if self._is_connected:
            logger.warning("MCPClient is already connected")
            return self
            
        try:
            if self._is_url:
                # For HTTP/SSE servers
                logger.debug(f"Connecting to MCP server via HTTP/SSE: {self.command_or_url}")
                
                # Wait for server to be ready
                if not await self._wait_for_server(self.command_or_url):
                    raise ConnectionError(f"Server at {self.command_or_url} is not responding")
                
                # Create the SSE connection with retry logic
                max_retries = 3
                retry_delay = 1.0
                
                for attempt in range(max_retries):
                    try:
                        streams = await self._exit_stack.enter_async_context(
                            sse_client(
                                self.command_or_url,
                                headers=self.headers
                            )
                        )
                        read_stream, write_stream = streams
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"SSE connection attempt {attempt + 1} failed: {e}. Retrying...")
                            await asyncio.sleep(retry_delay)
                        else:
                            raise ConnectionError(f"Failed to establish SSE connection after {max_retries} attempts: {e}")

            else:
                # For command-based servers (stdio)
                logger.debug(f"Connecting to MCP server: {self.command_or_url} {' '.join(self.args)}")
                
                # Create environment with defaults from current env
                env = {**os.environ, **self.env}
                
                # Set up server parameters
                server_params = StdioServerParameters(
                    command=self.command_or_url,
                    args=self.args,
                    env=env,
                    cwd=os.getcwd()
                )
                
                # Create the stdio connection
                streams = await self._exit_stack.enter_async_context(stdio_client(server_params))
                read_stream, write_stream = streams
            
            # Create the MCP session
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            # Initialize the session (longer timeout for stability)
            logger.debug("Initializing MCP session...")
            try:
                await asyncio.wait_for(
                    self._session.initialize(),
                    timeout=90.0  # Very generous timeout for initialization
                )
                logger.debug("MCP session initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing MCP session: {type(e).__name__} - {str(e)}")
                raise
            
            # Get available tools
            logger.debug("Fetching available MCP tools...")
            try:
                result = await asyncio.wait_for(self._session.list_tools(), timeout=30.0)
                self._mcp_tools = result.tools
                logger.debug(f"Found {len(self._mcp_tools)} MCP tools")
            except Exception as e:
                logger.error(f"Error getting MCP tools list: {type(e).__name__} - {str(e)}")
                raise
            
            self._is_connected = True
            logger.info(f"Successfully connected to MCP server: {self.command_or_url}")
            return self
            
        except Exception as e:
            # Ensure resources are cleaned up on connection failure
            await self.disconnect()
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        self._is_connected = False
        await self._exit_stack.aclose()
        self._session = None
        self._mcp_tools = []
    
    def is_connected(self) -> bool:
        """
        Check if the client is connected to an MCP server.
        
        Returns:
            bool: True if connected, False otherwise.
        """
        return self._is_connected and self._session is not None
    
    @asynccontextmanager
    async def connection(self) -> AsyncGenerator["MCPClient", None]:
        """
        Context manager for MCP server connection.
        
        Yields:
            MCPClient: The connected client instance.
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()
    
    def generate_function_from_mcp_tool(self, mcp_tool: "MCPTool") -> Callable:
        """
        Generate a Python function from an MCP tool definition.
        
        Args:
            mcp_tool (MCPTool): MCP tool to generate a function for.
            
        Returns:
            Callable: A Python function that calls the MCP tool.
        """
        func_name = mcp_tool.name
        func_desc = mcp_tool.description or f"Call the {func_name} MCP tool"
        
        input_schema = mcp_tool.inputSchema
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        # Generate function parameters and annotations
        func_params = list(properties.keys())
        annotations = {"return": Any}
        defaults = {}
        
        for param, schema in properties.items():
            param_type = schema.get("type", "Any")
            if param_type == "string":
                annotations[param] = str
            elif param_type == "number":
                annotations[param] = float
            elif param_type == "integer":
                annotations[param] = int
            elif param_type == "boolean":
                annotations[param] = bool
            elif param_type == "array":
                annotations[param] = List[Any]
            elif param_type == "object":
                annotations[param] = Dict[str, Any]
            else:
                annotations[param] = Any
                
            # Set defaults for optional parameters
            if param not in required:
                defaults[param] = None
        
        # Define the dynamic function
        async def dynamic_function(**kwargs):
            if not self._session:
                raise RuntimeError("MCP client is not connected")
                
            # Call the MCP tool
            result = await self._session.call_tool(func_name, kwargs)
            
            # Process the result
            try:
                content = result.result.content
                if content.type == "text":
                    return content.text
                elif content.type == "json":
                    return content.json
                elif content.type == "binary":
                    return f"Binary data received ({len(content.binary)} bytes)"
                elif content.type == "error":
                    return f"Error: {content.error}"
                elif content.type == "embedded_resource":
                    # Return resource information if available
                    if hasattr(content, "name") and content.name:
                        return f"Embedded resource: {content.name}"
                    return "Embedded resource received"
                else:
                    msg = f"Received content of type '{content.type}'"
                    return f"{msg} which is not fully supported yet."
            except (IndexError, AttributeError) as e:
                logger.error(f"Error processing content from MCP tool response: {e}")
                raise e
        
        # Update function metadata
        dynamic_function.__name__ = func_name
        dynamic_function.__doc__ = func_desc
        dynamic_function.__annotations__ = annotations
        
        # Set function signature
        sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name=param,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=defaults.get(param, inspect.Parameter.empty),
                    annotation=annotations[param],
                )
                for param in func_params
            ]
        )
        dynamic_function.__signature__ = sig
        
        return dynamic_function
    
    def get_available_tools(self) -> List[str]:
        """
        Get a list of available MCP tool names.
        
        Returns:
            List[str]: List of tool names available from the connected MCP server.
        """
        return [tool.name for tool in self._mcp_tools]
    
    def get_tool_info(self) -> dict:
        """
        Get information about this MCP client.
        
        Returns:
            dict: Information about the MCP client.
        """
        return {
            "name": "MCPClient",
            "description": "Client for connecting to and interacting with MCP servers",
            "connected": self.is_connected(),
            "server": self.command_or_url,
            "available_tools": self.get_available_tools() if self.is_connected() else []
        }


class MCPToolkit(Tool):
    """
    Toolkit for managing multiple MCP server connections.
    
    This class handles the lifecycle of multiple MCP server connections and
    offers a centralized way to access tools from multiple MCP servers.
    
    Args:
        servers (Optional[List[MCPClient]], optional): List of MCPClient
            instances to manage. Defaults to None.
        config_path (Optional[str], optional): Path to a JSON configuration
            file defining MCP servers. Defaults to None.
            
    Note:
        Either `servers` or `config_path` must be provided. If both are provided,
        servers from both sources will be combined.
        
    Example configuration file format:
    ```json
    {
        "mcpServers": {
            "server1": {
                "url": "https://example.com/mcp",
                "timeout": 30,
                "headers": {
                    "Authorization": "Bearer YOUR_TOKEN"
                }
            },
            "server2": {
                "command": "/path/to/local/mcp/server",
                "args": ["--option", "value"],
                "env": {
                    "KEY": "VALUE"
                }
            }
        }
    }
    ```
    """
    
    def __init__(
        self,
        servers: Optional[List[MCPClient]] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize the MCP toolkit."""
        super().__init__()
        
        if not servers and not config_path:
            raise ValueError("Either servers or config_path must be provided")
            
        self.servers: List[MCPClient] = servers or []
        
        if config_path:
            self.servers.extend(self._load_servers_from_config(config_path))
            
        self._connected = False
    
    def _load_servers_from_config(self, config_path: str) -> List[MCPClient]:
        """
        Load MCP server configurations from a JSON file.
        
        Args:
            config_path (str): Path to the JSON configuration file.
            
        Returns:
            List[MCPClient]: List of configured MCPClient instances.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in config file '{config_path}': {e}")
                    raise e
        except FileNotFoundError:
            logger.warning(f"Config file not found: '{config_path}'")
            raise
            
        all_servers = []
        mcp_servers = data.get("mcpServers", {})
        
        if not isinstance(mcp_servers, dict):
            logger.warning("'mcpServers' is not a dictionary, skipping...")
            return all_servers
            
        for name, cfg in mcp_servers.items():
            if not isinstance(cfg, dict):
                logger.warning(f"Configuration for server '{name}' must be a dictionary")
                continue
                
            if "command" not in cfg and "url" not in cfg:
                logger.warning(f"Missing required 'command' or 'url' field for server '{name}'")
                continue
                
            command_or_url = cfg.get("command") or cfg.get("url")
            
            server = MCPClient(
                command_or_url=command_or_url,
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                timeout=cfg.get("timeout"),
                headers=cfg.get("headers")
            )
            
            all_servers.append(server)
            
        return all_servers
    
    async def connect(self):
        """
        Connect to all MCP servers.
        
        Returns:
            MCPToolkit: The connected toolkit instance.
        """
        if self._connected:
            logger.warning("MCPToolkit is already connected")
            return self
            
        try:
            # Sequentially connect to each server
            for server in self.servers:
                await server.connect()
                
            self._connected = True
            return self
        except Exception as e:
            # Ensure resources are cleaned up on connection failure
            await self.disconnect()
            logger.error(f"Failed to connect to one or more MCP servers: {e}")
            raise e
    
    async def disconnect(self):
        """Disconnect from all MCP servers."""
        if not self._connected:
            return
            
        for server in self.servers:
            await server.disconnect()
            
        self._connected = False
    
    @asynccontextmanager
    async def connection(self) -> AsyncGenerator["MCPToolkit", None]:
        """
        Context manager for MCP server connections.
        
        Yields:
            MCPToolkit: The connected toolkit instance.
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()
    
    def is_connected(self) -> bool:
        """
        Check if all servers are connected.
        
        Returns:
            bool: True if all servers are connected, False otherwise.
        """
        return self._connected and all(server.is_connected() for server in self.servers)
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """
        Get available tools from all connected servers.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping server URLs to lists of available tool names.
        """
        tools = {}
        for server in self.servers:
            if server.is_connected():
                tools[server.command_or_url] = server.get_available_tools()
        return tools
    
    def get_all_tool_names(self) -> List[str]:
        """
        Get a flat list of all available tool names across all servers.
        
        Returns:
            List[str]: List of all available tool names.
        """
        all_tools = []
        for server in self.servers:
            if server.is_connected():
                all_tools.extend(server.get_available_tools())
        return all_tools
    
    def get_tool_info(self) -> dict:
        """
        Get information about this MCP toolkit.
        
        Returns:
            dict: Information about the MCP toolkit.
        """
        return {
            "name": "MCPToolkit",
            "description": "Toolkit for managing multiple MCP server connections",
            "connected": self.is_connected(),
            "servers": [server.command_or_url for server in self.servers],
            "available_tools": self.get_available_tools() if self.is_connected() else {}
        }
    
    