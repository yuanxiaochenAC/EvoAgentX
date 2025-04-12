import asyncio
import json
import os
import shlex
from typing import Optional, Dict, Any, List, Union, Set, Callable
import inspect
from contextlib import AsyncExitStack, asynccontextmanager
from urllib.parse import urlparse

from mcp import ClientSession, StdioServerParameters, Tool, ListToolsResult
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

class MCPClient:
    def __init__(
        self, 
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        # Initialize session and client objects
        self.command_or_url = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.headers = headers or {}
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._is_connected = False
        self._mcp_tools: List[Tool] = []
        
        
    @classmethod
    def from_config(cls, config: Dict[str, Any], server_name: Optional[str] = None) -> List['MCPClient']:
        """Create MCPClient instances from a configuration dictionary
        
        Args:
            config: Configuration dictionary with mcpServers section
            server_name: Optional name of a specific server to load
            
        Returns:
            List of MCPClient instances
        """
        clients = []
        mcp_servers = config.get("mcpServers", {})
        
        if not isinstance(mcp_servers, dict):
            raise ValueError("'mcpServers' must be a dictionary")
        
        # If server_name is provided, only load that specific server
        if server_name:
            if server_name not in mcp_servers:
                raise ValueError(f"Server '{server_name}' not found in configuration")
            
            server_configs = {server_name: mcp_servers[server_name]}
        else:
            server_configs = mcp_servers
            
        for name, cfg in server_configs.items():
            if not isinstance(cfg, dict):
                print(f"Configuration for server '{name}' must be a dictionary")
                continue

            if "command" not in cfg and "url" not in cfg:
                print(f"Missing required 'command' or 'url' field for server '{name}'")
                continue
                
            command_or_url = cfg.get("command") or cfg.get("url")
            
            client = cls(
                command=command_or_url,
                args=cfg.get("args", []),
                env={**os.environ, **cfg.get("env", {})},
                timeout=cfg.get("timeout", None),
                headers=cfg.get("headers", None),
            )
            clients.append(client)
            
        return clients
    
    @classmethod
    def from_config_file(cls, config_path: str, server_name: Optional[str] = None) -> List['MCPClient']:
        """Create MCPClient instances from a configuration file
        
        Args:
            config_path: Path to JSON configuration file
            server_name: Optional name of a specific server to load
            
        Returns:
            List of MCPClient instances
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in config file '{config_path}': {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: '{config_path}'")
            
        return cls.from_config(data, server_name)
        
    async def connect(self):
        """Connect to an MCP server using the configured settings
        
        Returns:
            self: The connected client
        """
        if self._is_connected:
            print("Server is already connected")
            return self
            
        if not self.command_or_url:
            raise ValueError("No command or URL specified")
            
        try:
            if urlparse(self.command_or_url).scheme in ("http", "https"):
                # SSE client for HTTP connections
                stdio_transport = await self.exit_stack.enter_async_context(
                    sse_client(
                        self.command_or_url,
                        headers=self.headers,
                    )
                )
            else:
                # Stdio client for command-line server
                command = self.command_or_url
                arguments = self.args
                
                if not self.args:
                    argv = shlex.split(command)
                    if not argv:
                        raise ValueError("Command is empty")
                    
                    command = argv[0]
                    arguments = argv[1:]
                    
                # Special case for npx on Windows
                if os.name == "nt" and command.lower() == "npx":
                    command = "npx.cmd"
                    
                server_params = StdioServerParameters(
                    command=command, 
                    args=arguments, 
                    env=self.env
                )
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            await self.session.initialize()
            
            # List available tools
            response = await self.session.list_tools()
            self._mcp_tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in self._mcp_tools])
            
            self._is_connected = True
            return self
        except Exception as e:
            # Clean up on failure
            await self.cleanup()
            print(f"Failed to connect to MCP server: {e}")
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e
    
    @asynccontextmanager
    async def connection(self):
        """Async context manager for MCP server connection
        
        Yields:
            MCPClient: Connected client instance
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.cleanup()
            
    async def list_tools(self) -> ListToolsResult:
        """Get the list of available tools from the server
        
        Returns:
            ListToolsResult: Result containing available tools
        """
        if not self.session:
            raise RuntimeError("Not connected to a server. Call connect() first.")
            
        return await self.session.list_tools()
        
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool with the given arguments
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool
            
        Returns:
            Result from the tool execution
        """
        if not self.session:
            raise ValueError("Not connected to a server. Call connect() first.")
            
        result = await self.session.call_tool(tool_name, tool_args)
        return result.content
        
    def generate_function_from_mcp_tool(self, mcp_tool: Tool) -> Callable:
        """Dynamically generates a Python callable function from an MCP tool
        
        Args:
            mcp_tool: The MCP tool definition
            
        Returns:
            Callable: A dynamically created Python function that wraps the MCP tool
        """
        schema = self._build_tool_schema(mcp_tool)
        
        func_name = mcp_tool.name
        func_desc = mcp_tool.description or "No description provided."
        parameters_schema = mcp_tool.inputSchema.get("properties", {})
        required_params = mcp_tool.inputSchema.get("required", [])
        
        # Create type mapping for parameter annotations
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        # Build function signature information
        annotations = {}
        defaults = {}
        param_names = []
        
        for param_name, param_schema in parameters_schema.items():
            param_type = param_schema.get("type", "Any")
            param_type = type_map.get(param_type, Any)
            
            annotations[param_name] = param_type
            if param_name not in required_params:
                defaults[param_name] = None
                
            param_names.append(param_name)
        
        # Create a synchronous function that handles the async call internally
        async def wrapper(**kwargs):
            """Auto-generated function for MCP Tool interaction.
            
            Args:
                kwargs: Keyword arguments corresponding to MCP tool parameters.
                
            Returns:
                The result returned by the MCP tool.
            """
            # Check for missing required parameters
            missing_params = set(required_params) - set(kwargs.keys())
            if missing_params:
                raise ValueError(f"Missing required parameters: {missing_params}")
                
            # Filter out any kwargs that aren't in the parameter list
            valid_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
            
            try:
                result = await self.call_tool(func_name, valid_kwargs)
                
                return result
            except Exception as e:
                error_msg = f"Failed to call MCP tool '{func_name}': {e}"
                print(error_msg)
                return {"error": error_msg}
        
        # Update function metadata
        wrapper.__name__ = func_name
        wrapper.__doc__ = func_desc
        wrapper.__annotations__ = annotations
        
        # Add signature if inspect module is available
        if inspect:
            try:
                sig = inspect.Signature(
                    parameters=[
                        inspect.Parameter(
                            name=param,
                            kind=inspect.Parameter.KEYWORD_ONLY,
                            default=defaults.get(param, inspect.Parameter.empty),
                            annotation=annotations[param],
                        )
                        for param in param_names
                    ]
                )
                wrapper.__signature__ = sig
            except Exception:
                # If creating signature fails, continue without it
                pass
        
        return wrapper
        
    def _build_tool_schema(self, mcp_tool: Tool) -> Dict[str, Any]:
        """Build an OpenAI-compatible tool schema from an MCP tool
        
        Args:
            mcp_tool: The MCP tool
            
        Returns:
            Dict containing the tool schema
        """
        input_schema = mcp_tool.inputSchema
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "No description provided.",
                "parameters": parameters,
            },
        }
        
    def get_tool_functions(self) -> List[Callable]:
        """Get a list of callable functions for all MCP tools
        
        Returns:
            List of callable functions
        """
        return [self.generate_function_from_mcp_tool(tool) for tool in self._mcp_tools]
        
    def get_openai_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all MCP tools
        
        Returns:
            List of tool schemas
        """
        return [self._build_tool_schema(tool) for tool in self._mcp_tools]

    def get_tools(self) -> List[Callable]:
        """Returns a list of function objects representing the tools in the toolkit.
        Maintains compatibility with the camel implementation while not using FunctionTool.
        
        Returns:
            List[Callable]: A list of callable functions
        """
        
        return [(
                self.generate_function_from_mcp_tool(mcp_tool),
                self._build_tool_schema(mcp_tool),
            )
            for mcp_tool in self._mcp_tools]

    async def cleanup(self):
        """Clean up resources"""
        self._is_connected = False
        await self.exit_stack.aclose()
        self.session = None
        
    def is_connected(self) -> bool:
        """Check if the client is connected
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self._is_connected and self.session is not None
        
    def tools_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about available tools
        
        Returns:
            List of tool information dictionaries
        """
        return [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema,
            "required_params": tool.inputSchema.get("required", []),
            "parameters": tool.inputSchema.get("properties", {})
        } for tool in self._mcp_tools]


class MCPToolkit:
    """A toolkit that manages multiple MCP server connections"""
    
    def __init__(
        self,
        servers: Optional[List[MCPClient]] = None,
        config_path: Optional[str] = None,
    ):
        if not servers and not config_path:
            raise ValueError("Either servers or config_path must be provided")
            
        if servers and config_path:
            print("Both servers and config_path are provided. Servers from both sources will be combined.")
            
        self.servers: List[MCPClient] = servers or []
        
        if config_path:
            self.servers.extend(MCPClient.from_config_file(config_path))
            
        self._connected = False
        self._tools_cache = {}  # Cache mapping tool names to server indices
            
    async def connect(self):
        """Connect to all servers
        
        Returns:
            MCPToolkit: The connected toolkit
        """
        if self._connected:
            print("MCPToolkit is already connected")
            return self
            
        try:
            # Connect to each server sequentially
            for i, server in enumerate(self.servers):
                await server.connect()
                
                # Cache the tools available on this server
                tools_info = server.tools_info()
                for tool in tools_info:
                    tool_name = tool["name"]
                    if tool_name not in self._tools_cache:
                        self._tools_cache[tool_name] = i
                
            self._connected = True
            return self
        except Exception as e:
            # Clean up on failure
            await self.disconnect()
            print(f"Failed to connect to one or more MCP servers: {e}")
            raise RuntimeError(f"Failed to connect to one or more MCP servers: {e}") from e
            
    async def disconnect(self):
        """Disconnect from all servers"""
        if not self._connected:
            return
            
        for server in self.servers:
            try:
                await server.cleanup()
            except Exception as e:
                # Suppress the specific cancel scope error
                if "cancel scope" not in str(e).lower():
                    print(f"Error disconnecting from server: {e}")
                
        self._connected = False
        self._tools_cache = {}  # Clear the cache on disconnect
        
    @asynccontextmanager
    async def connection(self):
        """Async context manager for toolkit connections
        
        Yields:
            MCPToolkit: Connected toolkit
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()
            
    def is_connected(self) -> bool:
        """Check if all servers are connected
        
        Returns:
            bool: True if all connected, False otherwise
        """
        return self._connected and all(server.is_connected() for server in self.servers)
        
    def get_all_openai_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all tools from all servers
        
        Returns:
            List of tool schemas
        """
        if not self.is_connected():
            raise RuntimeError("MCPToolkit is not connected. Call connect() first.")
            
        all_schemas = []
        for server in self.servers:
            if server.is_connected():
                all_schemas.extend(server.get_openai_tool_schemas())
                
        return all_schemas
        
    def get_tools(self) -> List[Callable]:
        """Aggregates all tools from the managed MCP server instances.
        Provides compatibility with the camel implementation.

        Returns:
            List[Callable]: Combined list of all available functions.

        Raises:
            RuntimeError: If the toolkit is not connected.
        """
        if not self.is_connected():
            raise RuntimeError("MCPToolkit is not connected. Call connection() first.")

        all_tools = []
        for server in self.servers:
            if server.is_connected():
                all_tools.extend(server.get_tools())
        return all_tools
        
    def detailed_tools_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get detailed information about all tools grouped by server
        
        Returns:
            Dictionary mapping server name/index to list of tool information
        """
        if not self.is_connected():
            raise RuntimeError("MCPToolkit is not connected. Call connect() first.")
            
        result = {}
        for i, server in enumerate(self.servers):
            if server.is_connected():
                server_name = f"server_{i}"
                result[server_name] = server.tools_info()
                
        return result
        
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool with the given arguments across all servers
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool
            
        Returns:
            Result from the tool execution
            
        Raises:
            ValueError: If not connected or if the tool is not found on any server
        """
        if not self.is_connected():
            raise ValueError("Not connected to any servers. Call connect() first.")
        
        # Use the cache to find the server with the requested tool
        if tool_name in self._tools_cache:
            server_index = self._tools_cache[tool_name]
            server = self.servers[server_index]
            
            if server.is_connected():
                result = await server.call_tool(tool_name, tool_args)
                return result
        
        # Fallback: search all servers if not in cache
        for i, server in enumerate(self.servers):
            if not server.is_connected():
                continue
                
            # Check if this server has the tool
            tools_info = server.tools_info()
            tool_names = [tool["name"] for tool in tools_info]
            
            if tool_name in tool_names:
                # Update the cache
                self._tools_cache[tool_name] = i
                
                # Call the tool on this server
                result = await server.call_tool(tool_name, tool_args)
                return result
                
        # If we get here, the tool wasn't found on any server
        raise ValueError(f"Tool '{tool_name}' not found on any connected server")
