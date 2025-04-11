"""
MCP Wrapper for EvoAgentX.

This module provides a simplified wrapper around the MCP Python SDK for use in EvoAgentX,
following patterns from the official MCP documentation.
"""

import os
import asyncio
import inspect
from typing import Dict, List, Any, Callable, Optional, AsyncContextManager
from contextlib import asynccontextmanager

try:
    from mcp import ClientSession, types
    from mcp.client.stdio import StdioServerParameters, stdio_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from ..core.logging import logger


class MCPWrapper:
    """
    Simple wrapper for MCP server connections following SDK patterns.
    
    This class provides a more straightforward interface to MCP servers,
    following patterns from the official MCP Python SDK documentation.
    
    Args:
        command (str): Command to launch the MCP server.
        args (List[str], optional): Command-line arguments.
        env (Dict[str, str], optional): Environment variables.
        cwd (str, optional): Working directory. Defaults to current directory.
    """
    
    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        """Initialize the MCP wrapper."""
        if not HAS_MCP:
            raise ImportError("MCP Python SDK not found. Please install it with 'pip install mcp'")
        
        self.command = command
        self.args = args or []
        self.env = {**os.environ, **(env or {})}
        self.cwd = cwd or os.getcwd()
        
        self._session = None
        self._tools_cache = None
    
    @asynccontextmanager
    async def session(self) -> AsyncContextManager[ClientSession]:
        """
        Create and manage an MCP session.
        
        This context manager handles connecting to the MCP server, creating a session,
        initializing it, and properly cleaning up resources.
        
        Yields:
            ClientSession: An initialized MCP client session.
        
        Example:
            ```python
            async with mcp_wrapper.session() as session:
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"Tool: {tool.name}")
            ```
        """
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
            cwd=self.cwd
        )
        
        logger.debug(f"Connecting to MCP server: {self.command} {' '.join(self.args)}")
        
        # First level: connect to the server
        async with stdio_client(server_params) as (read, write):
            logger.debug("Connection established")
            
            # Create session
            session = ClientSession(read, write)
            self._session = session
            
            try:
                # Initialize the session
                logger.debug("Initializing MCP session")
                await session.initialize()
                logger.debug("MCP session initialized")
                
                # Yield the initialized session
                yield session
            finally:
                logger.debug("Closing MCP session")
                self._session = None
                self._tools_cache = None
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of available tools from the MCP server.
        
        This method connects to the MCP server, gets the list of tools,
        and returns a simplified representation of each tool.
        
        Returns:
            List[Dict[str, Any]]: List of tool descriptions.
        
        Example:
            ```python
            tools = await mcp_wrapper.get_available_tools()
            for tool in tools:
                print(f"Tool: {tool['name']} - {tool['description']}")
            ```
        """
        async with self.session() as session:
            tools_result = await session.list_tools()
            
            # Convert tools to a simpler format
            tool_descriptions = []
            for tool in tools_result.tools:
                tool_descriptions.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
            
            return tool_descriptions
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool with the given arguments.
        
        Args:
            tool_name (str): Name of the tool to call.
            arguments (Dict[str, Any]): Arguments to pass to the tool.
            
        Returns:
            Any: The result of the tool call, processed to extract the content.
            
        Example:
            ```python
            result = await mcp_wrapper.call_tool(
                "github_search_repositories", 
                {"query": "language:python", "limit": 5}
            )
            print(f"Search results: {result}")
            ```
        """
        async with self.session() as session:
            logger.debug(f"Calling tool: {tool_name} with args: {arguments}")
            
            result = await session.call_tool(tool_name, arguments)
            
            # Extract the content based on type
            content = result.result.content
            if content.type == "text":
                return content.text
            elif content.type == "json":
                return content.json
            elif content.type == "binary":
                return content.binary
            elif content.type == "error":
                raise ValueError(f"Tool error: {content.error}")
            else:
                return f"Unsupported content type: {content.type}"
    
    @staticmethod
    def generate_function(
        tool_name: str, 
        description: str, 
        input_schema: Dict[str, Any]
    ) -> Callable:
        """
        Generate a Python function from an MCP tool definition.
        
        This static method creates a Python function that will call an MCP tool,
        with proper documentation and type hints based on the input schema.
        
        Args:
            tool_name (str): Name of the tool.
            description (str): Description of the tool.
            input_schema (Dict[str, Any]): JSON Schema for the tool's inputs.
            
        Returns:
            Callable: A Python function that takes the appropriate parameters.
        """
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        # Generate function parameters
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
        
        # Create docstring with parameter descriptions
        docstring = f"{description}\n\n"
        if properties:
            docstring += "Args:\n"
            for param, schema in properties.items():
                param_desc = schema.get("description", "")
                docstring += f"    {param}: {param_desc}\n"
        
        # Define the function
        def func(**kwargs):
            """Generated function that will be replaced with the docstring."""
            # This is a placeholder that will be used with the MCP tool
            pass
        
        # Set function metadata
        func.__name__ = tool_name
        func.__doc__ = docstring
        func.__annotations__ = annotations
        
        # Set function signature
        sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name=param,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=defaults.get(param, inspect.Parameter.empty),
                    annotation=annotations.get(param, Any),
                )
                for param in func_params
            ]
        )
        func.__signature__ = sig
        
        return func


# Example usage
async def main():
    """Example of using the MCPWrapper."""
    # GitHub MCP server example
    wrapper = MCPWrapper(
        command="npx",
        args=["--yes", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"}
    )
    
    try:
        # Get available tools
        tools = await wrapper.get_available_tools()
        print(f"Found {len(tools)} tools:")
        
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
            
            # Generate a function for this tool
            func = MCPWrapper.generate_function(
                tool['name'],
                tool['description'],
                tool['input_schema']
            )
            
            # Display function signature
            print(f"    Function: {func.__name__}{inspect.signature(func)}")
        
        # Call a tool (example)
        if tools:
            sample_tool = tools[0]
            print(f"\nCalling tool: {sample_tool['name']}")
            # You would call it like this:
            # result = await wrapper.call_tool(sample_tool['name'], {})
            # print(f"Result: {result}")
    
    except Exception as e:
        print(f"Error: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 