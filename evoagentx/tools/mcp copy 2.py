"""
This tool is inspired by MCP Python SDK and MCP toolkit from the Camel-AI project. You may find more information about by visiting the following links:
- https://github.com/modelcontextprotocol/python-sdk
- https://github.com/camel-ai/camel/blob/master/camel/toolkits/mcp_toolkit.py

"""
import threading
import asyncio
from functools import partial
from typing import Optional, Dict, Any, List, Callable, AsyncGenerator
from evoagentx.tools.tool import Tool
from evoagentx.core.logging import logger
from contextlib import AsyncExitStack, asynccontextmanager
from urllib.parse import urlparse
import mcp
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client


class Adapter:
    def adapt(
        func: Callable[[dict | None], mcp.types.CallToolResult],
        mcp_tool: mcp.types.Tool,
    ) -> Tool:
        
        class MCPAdaptTool(Tool):
            def __init__(self, name: str, description: str, schema: dict[str, Any]):
                self.name = name
                self.descriptions = [description]
                self.schemas = [schema]
                self.tools = [func]
            
        input_schema = mcp_tool.inputSchema
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
                "description": mcp_tool.description or "No description provided.",
                "parameters": parameters,
            },
        }

        print(f"Creating tool: {mcp_tool.name}")
        print(f"Tool description: {mcp_tool.description or ''}")
        print(f"Tool input schema: {input_schema}")
        print(f"tools: {[func]}")
        
        tool = MCPAdaptTool(
            name=mcp_tool.name,
            description=mcp_tool.description,
            schema=schema,
        )

        return tool


class MCPClient:
    
    def __init__(self, 
                 server_configs: StdioServerParameters | dict[str, Any] | list[StdioServerParameters | dict[str, Any]],
                 connect_timeout: float = 120.0,
                 adapter: Adapter = Adapter,
                 ):
        
        if isinstance(server_configs, list):
            self.server_configs = server_configs
        else:
            self.server_configs = [server_configs]
        
        self.event_loop = asyncio.new_event_loop()
        
        self.sessions:list[mcp.ClientSession] = []
        self.mcp_tools:list[list[mcp.types.Tool]] = []
        
        self.task = None
        self.adapter = adapter
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
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._disconnect()
    
    def _run_event(self):
        """Runs the event loop in a separate thread (for synchronous usage)."""
        print("Running event loop")
        asyncio.set_event_loop(self.event_loop)

        async def setup():
            async with AsyncExitStack() as stack:
                connections = [
                    await stack.enter_async_context(self._start_server(params))
                    for params in self.server_configs
                ]
                self.sessions, self.mcp_tools = [list(c) for c in zip(*connections)]
                self.thread_running.set()  # Signal initialization is complete
                await asyncio.Event().wait()  # Keep session alive until stopped

        asyncio.run(setup())
        self.task = self.event_loop.create_task(setup())
        
        try:
            self.event_loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            pass

    @asynccontextmanager
    async def _start_server(self, server_config: StdioServerParameters | dict[str, Any]):
        if isinstance(server_config, StdioServerParameters):
            client = stdio_client(server_config)
        elif isinstance(server_config, dict):
            client = sse_client(**server_config)
        else:
            raise ValueError(f"Invalid server config type: {type(server_config)}")
        
        async with client as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection and get the tools from the mcp server
                await session.initialize()
                tools = await session.list_tools()
                yield session, tools.tools

    def get_tools(self) -> List[Tool]:
        if not self.sessions:
            raise RuntimeError("Session not initialized")

        def _sync_call_tool(
            session, name: str, arguments: dict | None = None
        ) -> mcp.types.CallToolResult:
            return asyncio.run_coroutine_threadsafe(
                session.call_tool(name, arguments), self.event_loop
            ).result()

        return [
            self.adapter.adapt(func=partial(_sync_call_tool, session, tool.name), mcp_tool=tool)
            for session, tools in zip(self.sessions, self.mcp_tools)
            for tool in tools
        ]
    
    
class MCPToolkit:
    def __init__(self, servers: Optional[list[MCPClient]] = None, config_path: Optional[str] = None, config: Optional[dict[str, Any]] = None):
        self.servers: list[MCPClient] = servers
        
        parameters = []
        if config_path:
            parameters += self._from_config_file(config_path)
        if config:
            parameters += self._from_config(config)
        
        self.servers = [MCPClient(parameters)]
        for server in self.servers:
            server._connect()
    
    def _from_config_file(self, config_path: str):
        with open(config_path, "r") as f:
            server_configs = json.load(f)
        return self._from_config(server_configs)
    
    def _from_config(self, server_configs: dict[str, Any]):
        parameters = []
        if server_configs.get("mcpServers", None):
            server_configs = server_configs.get("mcpServers")
        for name, cfg in server_configs.items():
            if not isinstance(cfg, dict):
                print(f"Configuration for server '{name}' must be a dictionary")
                continue

            if "command" not in cfg and "url" not in cfg:
                print(f"Missing required 'command' or 'url' field for server '{name}'")
                continue
                
            command_or_url = cfg.get("command") or cfg.get("url")
            parameters.append(StdioServerParameters(command=command_or_url, args=cfg.get("args", []), env={**os.environ, **cfg.get("env", {})}, timeout=cfg.get("timeout", None), headers=cfg.get("headers", None)))
            
        return parameters
        
    def get_tools(self):
        """Return a flattened list of all tools across all servers"""
        all_tools = []
        for server in self.servers:
            all_tools.extend(server.get_tools())
        return all_tools
