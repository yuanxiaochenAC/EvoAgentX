"""
This tool is inspired / modified from MCP Python SDK and mcpadapt projects, now enhanced with FastMCP 2.0. 
You may find more information about by visiting the following links:
- https://github.com/modelcontextprotocol/python-sdk
- https://github.com/grll/mcpadapt
- https://gofastmcp.com/clients/client

FastMCP 2.0 Integration Notes:
- Replaced official MCP SDK with FastMCP for better performance and reliability
- Maintains the same synchronous API with internal async handling via threading
- Auto-infers transport types (stdio/HTTP) based on configuration
- Enhanced error handling with FastMCP's exception hierarchy
- Backwards compatible with existing MCP server configurations
"""
import threading
import asyncio
from pydantic import Field
from functools import partial
from typing import Optional, Any, List, Dict, Callable
from evoagentx.tools.tool import Tool
from evoagentx.core.logging import logger
from contextlib import AsyncExitStack, asynccontextmanager

# FastMCP 2.0 imports - replacing official MCP SDK
from fastmcp import Client
from fastmcp.exceptions import ClientError, McpError
from fastmcp.client.transports import PythonStdioTransport, UvxStdioTransport, StreamableHttpTransport
import os
import json

# Keep StdioServerParameters for config compatibility
class StdioServerParameters:
    def __init__(self, command: str, args: List[str] = None, env: Dict[str, str] = None, 
                 timeout: Optional[float] = None, headers: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.headers = headers

class MCPTool(Tool):

    descriptions: List[str] = Field(default_factory=list, description="A list of descriptions for the tool")
    schemas: List[dict[str, Any]] = Field(default_factory=list, description="A list of schemas for the tool")
    tools: List[Callable] = Field(default_factory=list, description="A list of tools for the tool")

    def __init__(
        self,
        name: str = "MCPTool",
        descriptions: List[str] = ["Default MCP Tool description"],
        schemas: List[dict[str, Any]] = [],
        tools: List[Callable] = [],
    ):
        super().__init__(
            name=name,
            descriptions=descriptions,
            schemas=schemas,
            tools=tools
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return self.schemas
    
    def get_tool_descriptions(self) -> List[str]:
        return self.descriptions
    
    def get_tools(self) -> List[Callable]:
        return self.tools


class MCPClient:
    
    def __init__(
        self, 
        server_configs: StdioServerParameters | dict[str, Any] | list[StdioServerParameters | dict[str, Any]],
        connect_timeout: float = 120.0,
    ):
        
        if isinstance(server_configs, list):
            self.server_configs = server_configs
        else:
            self.server_configs = [server_configs]
        
        self.event_loop = asyncio.new_event_loop()
        
        self.sessions: list[Client] = []
        self.mcp_tools: list[list[Any]] = []  # FastMCP tools
        
        self.task = None
        self.thread_running = threading.Event()
        ## Testing
        self.working_thread = threading.Thread(target=self._run_event, daemon=True)
        self.connect_timeout = connect_timeout
        
        self.tools = None
        self.tool_schemas = None
        self.tool_descriptions = None
    
    def _disconnect(self):
        if self.task and not self.task.done():
            self.event_loop.call_soon_threadsafe(self.task.cancel)
        self.working_thread.join()
        self.event_loop.close()
    
    def _connect(self):
        self.working_thread.start()
        # check connection to mcp server is ready
        if not self.thread_running.wait(timeout=self.connect_timeout):
            raise TimeoutError(
                f"Couldn't connect to the MCP server after {self.connect_timeout} seconds"
            )
    
    def __enter__(self):
        self.working_thread.start()
        # check connection to mcp server is ready
        if not self.thread_running.wait(timeout=self.connect_timeout):
            raise TimeoutError(
                f"Couldn't connect to the MCP server after {self.connect_timeout} seconds"
            )
        return self.get_tools()
    
    def __del__(self):
        self._disconnect()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._disconnect()
    
    def _run_event(self):
        """Runs the event loop in a separate thread (for synchronous usage)."""
        print("Running event loop")
        asyncio.set_event_loop(self.event_loop)

        async def setup():
            try:
                async with AsyncExitStack() as stack:
                    connections = [
                        await stack.enter_async_context(self._start_server(params))
                        for params in self.server_configs
                    ]
                    self.sessions, self.mcp_tools = [list(c) for c in zip(*connections)]
                    self.thread_running.set()  # Signal initialization is complete
                    
                    # Use asyncio.Future instead of Event for better cleanup
                    done_future = self.event_loop.create_future()
                    await done_future
            except Exception as e:
                logger.error(f"Error in MCP event loop: {str(e)}")
                self.thread_running.set()  # Still set the event so we don't hang
                raise

        # Create and run the task directly in the event loop
        self.task = self.event_loop.create_task(setup())
        
        try:
            self.event_loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            logger.info("MCP client event loop was cancelled")
        except Exception as e:
            logger.error(f"Error in MCP event loop: {str(e)}")

    @asynccontextmanager
    async def _start_server(self, server_config: StdioServerParameters | dict[str, Any]):
        # Create FastMCP transport based on command type
        if isinstance(server_config, StdioServerParameters):
            # Check if this is a uvx command
            if server_config.command == "uvx":
                # Use UvxStdioTransport for uvx commands
                tool_name = server_config.args[0] if server_config.args else "unknown-tool"
                transport = UvxStdioTransport(tool_name=tool_name)
            else:
                # For Python scripts, use PythonStdioTransport with script_path
                transport = PythonStdioTransport(
                    script_path=server_config.command,
                    args=server_config.args,
                    env=server_config.env
                )
            client = Client(transport)
        elif isinstance(server_config, dict):
            if "url" in server_config:
                # For HTTP/SSE, use StreamableHttpTransport
                transport = StreamableHttpTransport(url=server_config["url"])
                client = Client(transport)
            else:
                # For stdio commands from dict config
                command = server_config.get("command", "")
                args = server_config.get("args", [])
                env = {**os.environ, **server_config.get("env", {})}
                
                if command == "uvx":
                    # Use UvxStdioTransport for uvx commands
                    tool_name = args[0] if args else "unknown-tool"
                    transport = UvxStdioTransport(tool_name=tool_name)
                else:
                    # For Python scripts, use PythonStdioTransport with script_path
                    transport = PythonStdioTransport(
                        script_path=command,
                        args=args,
                        env=env
                    )
                client = Client(transport)
        else:
            raise ValueError("Invalid server config type: {}".format(type(server_config)))
        
        async with client:
            # FastMCP Client handles initialization automatically
            tools = await client.list_tools()
            yield client, tools

    def create_tool(
        self,
        function: Callable[[dict | None], Any],
        mcp_tool: Any,  # FastMCP tool object
    ) -> MCPTool:

        # FastMCP tools should have similar structure to official MCP
        input_schema = getattr(mcp_tool, 'inputSchema', {})
        if not input_schema and hasattr(mcp_tool, 'input_schema'):
            input_schema = mcp_tool.input_schema
            
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        schema = {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": getattr(mcp_tool, 'description', None) or "No description provided.",
                "parameters": parameters,
            },
        }
        
        tool = MCPTool(
            name=mcp_tool.name,
            descriptions=[getattr(mcp_tool, 'description', None) or ""],
            schemas=[schema],
            tools=[function],
        )

        return tool
    
    def get_tools(self) -> List[Tool]:
        if not self.sessions:
            raise RuntimeError("Session not initialized")

        def _sync_call_tool(
            session, name: str, **kwargs
        ) -> Any:
            try:
                # Handle both direct parameters and arguments-wrapped parameters
                if "arguments" in kwargs and len(kwargs) == 1:
                    # Original format: {"arguments": {...}}
                    arguments = kwargs["arguments"]
                else:
                    # New format: direct parameters
                    arguments = kwargs
                
                logger.info(f"Calling MCP tool: {name} with arguments: {arguments}")
                # FastMCP Client's call_tool method
                future = asyncio.run_coroutine_threadsafe(
                    session.call_tool(name, arguments), self.event_loop
                )
                result = future.result(timeout=60)  # Increased timeout from 15 to 60 seconds
                logger.info(f"MCP tool {name} call completed successfully")
                return result
            except (TimeoutError, ClientError, McpError) as e:
                logger.error(f"Error calling MCP tool {name}: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error calling MCP tool {name}: {str(e)}")
                raise

        return [
            self.create_tool(partial(_sync_call_tool, session, tool.name), tool)
            for session, tools in zip(self.sessions, self.mcp_tools)
            for tool in tools
        ]
    
    
class MCPToolkit:
    def __init__(self, servers: Optional[list[MCPClient]] = None, config_path: Optional[str] = None, config: Optional[dict[str, Any]] = None):
        
        parameters = []
        if config_path:
            parameters += self._from_config_file(config_path)
        if config:
            parameters += self._from_config(config)
        
        self.servers = [MCPClient(parameters)]
        if servers:
            self.servers += servers
        for server in self.servers:
            try:
                server._connect()
                logger.info("Successfully connected to MCP servers")
            except TimeoutError as e:
                logger.warning(f"Timeout connecting to MCP servers: {str(e)}. Some tools may not be available.")
            except Exception as e:
                logger.error(f"Error connecting to MCP servers: {str(e)}")
    
    def _from_config_file(self, config_path: str):
        try:
            with open(config_path, "r") as f:
                server_configs = json.load(f)
            return self._from_config(server_configs)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file: {config_path}")
            return []
    
    def _from_config(self, server_configs: dict[str, Any]):
        parameters = []
        if server_configs.get("mcpServers", None):
            server_configs = server_configs.get("mcpServers")
        for name, cfg in server_configs.items():
            if not isinstance(cfg, dict):
                logger.warning(f"Configuration for server '{name}' must be a dictionary")
                continue

            if "command" not in cfg and "url" not in cfg:
                logger.warning(f"Missing required 'command' or 'url' field for server '{name}'")
                continue
                
            # For FastMCP 2.0, we can pass the config dict directly or create StdioServerParameters
            if "url" in cfg:
                # HTTP/SSE transport - pass dict directly for FastMCP auto-inference
                parameters.append(cfg)
                logger.info(f"Configured FastMCP HTTP/SSE server: {name} with URL: {cfg['url']}")
            else:
                # Stdio transport - create StdioServerParameters for compatibility
                command_or_url = cfg.get("command")
                timeout = cfg.get("timeout", None)
                parameters.append(StdioServerParameters(
                    command=command_or_url, 
                    args=cfg.get("args", []), 
                    env={**os.environ, **cfg.get("env", {})}, 
                    timeout=timeout, 
                    headers=cfg.get("headers", None)
                ))
                logger.info(f"Configured FastMCP stdio server: {name} with command: {command_or_url}")
            
        return parameters
    
    def disconnect(self):
        for server in self.servers:
            server._disconnect()
        
    def get_tools(self):
        """Return a flattened list of all tools across all servers"""
        all_tools = []
        for server in self.servers:
            try:
                tools = server.get_tools()
                all_tools.extend(tools)
                logger.info(f"Added {len(tools)} tools from MCP server")
            except Exception as e:
                logger.error(f"Error getting tools from MCP server: {str(e)}")
        return all_tools
