# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import inspect
import json
import os
import shlex
from contextlib import AsyncExitStack, asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    cast,
)
from urllib.parse import urlparse

if TYPE_CHECKING:
    from mcp import ClientSession, ListToolsResult, Tool

from ..core.logging import logger
from .tool import BaseToolkit, FunctionTool

logger = get_logger(__name__)


class MCPClient(BaseToolkit):
    r"""Internal class that provides an abstraction layer to interact with
    external tools using the Model Context Protocol (MCP). It supports two
    modes of connection:

    1. stdio mode: Connects via standard input/output streams for local
       command-line interactions.

    2. SSE mode (HTTP Server-Sent Events): Connects via HTTP for persistent,
       event-based interactions.

    Attributes:
        command_or_url (str): URL for SSE mode or command executable for stdio
            mode. (default: :obj:`'None'`)
        args (List[str]): List of command-line arguments if stdio mode is used.
            (default: :obj:`'None'`)
        env (Dict[str, str]): Environment variables for the stdio mode command.
            (default: :obj:`'None'`)
        timeout (Optional[float]): Connection timeout. (default: :obj:`'None'`)
        headers (Dict[str, str]): Headers for the HTTP request.
            (default: :obj:`'None'`)
    """

    def __init__(
        self,
        command_or_url: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        from mcp import Tool

        super().__init__(timeout=timeout)

        self.command_or_url = command_or_url
        self.args = args or []
        self.env = env or {}
        self.headers = headers or {}

        self._mcp_tools: List[Tool] = []
        self._session: Optional['ClientSession'] = None
        self._exit_stack = AsyncExitStack()
        self._is_connected = False

    async def connect(self):
        r"""Explicitly connect to the MCP server.

        Returns:
            MCPClient: The client used to connect to the server.
        """
        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client
        from mcp.client.stdio import StdioServerParameters, stdio_client

        if self._is_connected:
            logger.warning("Server is already connected")
            return self

        try:
            if urlparse(self.command_or_url).scheme in ("http", "https"):
                (
                    read_stream,
                    write_stream,
                ) = await self._exit_stack.enter_async_context(
                    sse_client(
                        self.command_or_url,
                        headers=self.headers,
                    )
                )
            else:
                command = self.command_or_url
                arguments = self.args
                if not self.args:
                    argv = shlex.split(command)
                    if not argv:
                        raise ValueError("Command is empty")

                    command = argv[0]
                    arguments = argv[1:]

                if os.name == "nt" and command.lower() == "npx":
                    command = "npx.cmd"

                server_parameters = StdioServerParameters(
                    command=command, args=arguments, env=self.env
                )
                (
                    read_stream,
                    write_stream,
                ) = await self._exit_stack.enter_async_context(
                    stdio_client(server_parameters)
                )

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            list_tools_result = await self.list_mcp_tools()
            self._mcp_tools = list_tools_result.tools
            self._is_connected = True
            return self
        except Exception as e:
            # Ensure resources are cleaned up on connection failure
            await self.disconnect()
            logger.error(f"Failed to connect to MCP server: {e}")
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

    async def disconnect(self):
        r"""Explicitly disconnect from the MCP server."""
        self._is_connected = False
        await self._exit_stack.aclose()
        self._session = None

    @asynccontextmanager
    async def connection(self):
        r"""Async context manager for establishing and managing the connection
        with the MCP server. Automatically selects SSE or stdio mode based
        on the provided `command_or_url`.

        Yields:
            MCPClient: Instance with active connection ready for tool
                interaction.
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()

    async def list_mcp_tools(self) -> Union[str, "ListToolsResult"]:
        r"""Retrieves the list of available tools from the connected MCP
        server.

        Returns:
            ListToolsResult: Result containing available MCP tools.
        """
        if not self._session:
            raise RuntimeError("MCP Client is not connected. Call `connection()` first.")
        try:
            return await self._session.list_tools()
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            raise RuntimeError(f"Failed to list MCP tools: {e}") from e

    def generate_function_from_mcp_tool(self, mcp_tool: "Tool") -> Callable:
        r"""Dynamically generates a Python callable function corresponding to
        a given MCP tool.

        Args:
            mcp_tool (Tool): The MCP tool definition received from the MCP
                server.

        Returns:
            Callable: A dynamically created async Python function that wraps
                the MCP tool.
        """
        func_name = mcp_tool.name
        func_desc = mcp_tool.description or "No description provided."
        parameters_schema = mcp_tool.inputSchema.get("properties", {})
        required_params = mcp_tool.inputSchema.get("required", [])

        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        annotations = {}  # used to type hints
        defaults: Dict[str, Any] = {}  # store default values

        func_params = []
        for param_name, param_schema in parameters_schema.items():
            param_type = param_schema.get("type", "Any")
            param_type = type_map.get(param_type, Any)

            annotations[param_name] = param_type
            if param_name not in required_params:
                defaults[param_name] = None

            func_params.append(param_name)

        async def dynamic_function(**kwargs):
            r"""Auto-generated function for MCP Tool interaction.

            Args:
                kwargs: Keyword arguments corresponding to MCP tool parameters.

            Returns:
                str: The textual result returned by the MCP tool.
            """
            from mcp.types import CallToolResult

            missing_params: Set[str] = set(required_params) - set(
                kwargs.keys()
            )
            if missing_params:
                raise ValueError(f"Missing required parameters: {missing_params}")

            if not self._session:
                raise RuntimeError(
                    "MCP Client is not connected. Call `connection()` first."
                )

            try:
                result: CallToolResult = await self._session.call_tool(
                    func_name, kwargs
                )
            except Exception as e:
                logger.error(f"Failed to call MCP tool '{func_name}': {e}")
                raise RuntimeError(f"Failed to call MCP tool '{func_name}': {e}") from e

            if not result.content or len(result.content) == 0:
                return "No data available for this request."

            # Handle different content types
            try:
                content = result.content[0]
                if content.type == "text":
                    return content.text
                elif content.type == "image":
                    # Return image URL or data URI if available
                    if hasattr(content, "url") and content.url:
                        return f"Image available at: {content.url}"
                    return "Image content received (data URI not shown)"
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
                raise RuntimeError(f"Error processing content from MCP tool response: {e}") from e

        dynamic_function.__name__ = func_name
        dynamic_function.__doc__ = func_desc
        dynamic_function.__annotations__ = annotations

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
        dynamic_function.__signature__ = sig  # type: ignore[attr-defined]

        return dynamic_function

    def _build_tool_schema(self, mcp_tool: "Tool") -> Dict[str, Any]:
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
                "description": mcp_tool.description
                or "No description provided.",
                "parameters": parameters,
            },
        }

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        functions in the toolkit. Each function is dynamically generated
        based on the MCP tool definitions received from the server.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects
                representing the functions in the toolkit.
        """
        return [
            FunctionTool(
                self.generate_function_from_mcp_tool(mcp_tool),
                openai_tool_schema=self._build_tool_schema(mcp_tool),
            )
            for mcp_tool in self._mcp_tools
        ]

    @property
    def session(self) -> Optional["ClientSession"]:
        return self._session


class MCPToolkit(BaseToolkit):
    r"""MCPToolkit provides a unified interface for managing multiple
    MCP server connections and their tools.

    This class handles the lifecycle of multiple MCP server connections and
    offers a centralized configuration mechanism for both local and remote
    MCP services.

    Args:
        servers (Optional[List[MCPClient]]): List of MCPClient
            instances to manage.
        config_path (Optional[str]): Path to a JSON configuration file
            defining MCP servers.

    Note:
        Either `servers` or `config_path` must be provided. If both are
        provided, servers from both sources will be combined.

        For web servers in the config file, you can specify authorization
        headers using the "headers" field to connect to protected MCP server
        endpoints.

        Example configuration:

        .. code-block:: json

            {
              "mcpServers": {
                "protected-server": {
                  "url": "https://example.com/mcp",
                  "timeout": 30,
                  "headers": {
                    "Authorization": "Bearer YOUR_TOKEN",
                    "X-API-Key": "YOUR_API_KEY"
                  }
                }
              }
            }

    Attributes:
        servers (List[MCPClient]): List of MCPClient instances being managed.
    """

    def __init__(
        self,
        servers: Optional[List[MCPClient]] = None,
        config_path: Optional[str] = None,
    ):
        super().__init__()

        if not servers and not config_path:
            raise ValueError("Either servers or config_path must be provided")

        if servers and config_path:
            logger.warning(
                "Both servers and config_path are provided. "
                "Servers from both sources will be combined."
            )

        self.servers: List[MCPClient] = servers or []

        if config_path:
            self.servers.extend(self._load_servers_from_config(config_path))

        self._exit_stack = AsyncExitStack()
        self._connected = False

    def _load_servers_from_config(self, config_path: str) -> List[MCPClient]:
        r"""Loads MCP server configurations from a JSON file.

        Args:
            config_path (str): Path to the JSON configuration file.

        Returns:
            List[MCPClient]: List of configured MCPClient instances.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the config file contains invalid JSON.
            ValueError: If the config file has an invalid structure.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in config file '{config_path}': {e}")
                    raise
        except FileNotFoundError as e:
            logger.error(f"Config file not found: '{config_path}'")
            raise

        all_servers = []

        mcp_servers = data.get("mcpServers", {})
        if not isinstance(mcp_servers, dict):
            logger.error("'mcpServers' is not a dictionary")
            raise ValueError("'mcpServers' must be a dictionary")

        for name, cfg in mcp_servers.items():
            if not isinstance(cfg, dict):
                logger.warning(
                    f"Configuration for server '{name}' must be a dictionary"
                )
                continue

            if "command" not in cfg and "url" not in cfg:
                logger.warning(
                    f"Missing required 'command' or 'url' field for server "
                    f"'{name}'"
                )
                continue

            server = MCPClient(
                command_or_url=cast(str, cfg.get("command") or cfg.get("url")),
                args=cfg.get("args", []),
                env={**os.environ, **cfg.get("env", {})},
                timeout=cfg.get("timeout", None),
                headers=cfg.get("headers", None),
            )
            all_servers.append(server)

        return all_servers

    async def connect(self):
        r"""Explicitly connect to all MCP servers.

        Returns:
            MCPToolkit: The connected toolkit instance

        Raises:
            RuntimeError: If connection to any server fails.
        """
        if self._connected:
            logger.warning("MCPToolkit is already connected")
            return self

        self._exit_stack = AsyncExitStack()
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
            raise RuntimeError(f"Failed to connect to one or more MCP servers: {e}") from e

    async def disconnect(self):
        r"""Explicitly disconnect from all MCP servers."""
        if not self._connected:
            return

        for server in self.servers:
            try:
                await server.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from server: {e}")

        self._connected = False
        await self._exit_stack.aclose()

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator["MCPToolkit", None]:
        r"""Async context manager that simultaneously establishes connections
        to all managed MCP server instances.

        Yields:
            MCPToolkit: Self with all servers connected.
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()

    def is_connected(self) -> bool:
        r"""Checks if all the managed servers are connected.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self._connected and all(server.is_connected() for server in self.servers)

    def get_tools(self) -> List[FunctionTool]:
        r"""Aggregates all tools from the managed MCP server instances.

        Returns:
            List[FunctionTool]: Combined list of all available function tools.

        Raises:
            RuntimeError: If the toolkit is not connected.
        """
        if not self.is_connected():
            raise RuntimeError("MCPToolkit is not connected. Call `connection()` first.")

        all_tools = []
        for server in self.servers:
            if server.is_connected():
                all_tools.extend(server.get_tools())
        return all_tools