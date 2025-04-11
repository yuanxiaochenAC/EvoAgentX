import asyncio
import os
import sys
import json
import tempfile
from pathlib import Path
import pprint
from openai import OpenAI

# Add the parent directory to sys.path to import from evoagentx
sys.path.insert(0, str(Path(__file__).parent.parent))

from evoagentx.tools.mcp import MCPClient, MCPToolkit
from evoagentx.models.model_configs import OpenAILLMConfig
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.core.registry import MODEL_REGISTRY
from evoagentx.agents.agent import Agent
from evoagentx.actions.action import Action, ActionInput, ActionOutput
# Sample config for GitHub MCP server
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

async def sample_direct_connection():
    """Sample that demonstrates connecting directly to a weather server"""
    print("\n==== DIRECT CONNECTION SAMPLE ====")
    
    # Get the path to the weather server script
    server_script_path = os.path.join(os.path.dirname(__file__), "weather_server.py")
    print(f"Weather server path: {server_script_path}")
    
    # Initialize the client with the direct script path
    client = MCPClient(command_or_url="python", args=[server_script_path])
    try:
        print("Connecting to weather server...")
        # Connect to the server
        await client.connect()
        
        # Get detailed information about available tools
        tools_info = client.tools_info()
        
        print(f"\n=== Found {len(tools_info)} tools ===")
        for i, tool in enumerate(tools_info):
            print(f"\nTOOL {i+1}: {tool['name']}")
            print(f"  Description: {tool['description']}")
            print(f"  Required parameters: {tool['required_params']}")
            print("  Parameters:")
            for param_name, param_schema in tool['parameters'].items():
                param_type = param_schema.get('type', 'any')
                param_desc = param_schema.get('description', 'No description')
                required = "Required" if param_name in tool['required_params'] else "Optional"
                print(f"    - {param_name} ({param_type}, {required}): {param_desc}")
        
        # Get OpenAI-compatible tool schemas
        openai_schemas = client.get_openai_tool_schemas()
        print("\n=== OpenAI Tool Schemas ===")
        for i, schema in enumerate(openai_schemas):
            print(f"\nSchema {i+1}: {schema['function']['name']}")
            print(f"  Description: {schema['function']['description']}")
            required = schema['function']['parameters'].get('required', [])
            print(f"  Required parameters: {required}")
        
        # Get dynamic functions for tools
        functions = client.get_tool_functions()
        print("\n=== Dynamic Tool Functions ===")
        for i, func in enumerate(functions):
            print(f"\nFunction {i+1}: {func.__name__}")
            print(f"  Doc: {func.__doc__}")
            print(f"  Signature: {inspect.signature(func)}")
        
        # Example of a direct tool call
        print("\n=== Direct Tool Call Example ===")
        print("Calling get_forecast tool for Sacramento coordinates...")
        try:
            # Sacramento coordinates
            result = await client.call_tool("get_forecast", {"latitude": 38.5816, "longitude": -121.4944})
            # Print the result type and first 200 characters for preview
            result_type = type(result).__name__
            result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            print(f"Result type: {result_type}")
            print(f"Result preview: {result_preview}")
        except Exception as e:
            print(f"Error calling tool: {e}")
    
    finally:
        print("\nCleaning up...")
        await client.cleanup()

async def search_repo_info(client, repo_owner, repo_name):
    """Search for repository information and display details"""
    print(f"\n=== Searching for repository: {repo_owner}/{repo_name} ===")
    try:
        # Use the search_repositories tool to find the repository
        search_query = f"repo:{repo_owner}/{repo_name}"
        search_result = await client.call_tool("search_repositories", {"query": search_query})
        
        # Display search result information
        if not search_result or not isinstance(search_result, list) or not search_result[0]:
            print("No repository found or unexpected result format.")
            return
            
        content = search_result[0]
        if hasattr(content, 'text'):
            # Print basic repository information
            result_text = content.text[:1000]  # Limit output size
            print("\nRepository search result:")
            print(result_text)
            
            # Try to get more details about the repository if content looks like JSON
            try:
                if "{" in content.text:
                    repo_data = json.loads(content.text)
                    if isinstance(repo_data, dict) and repo_data.get("items") and len(repo_data["items"]) > 0:
                        repo = repo_data["items"][0]
                        print("\nRepository details:")
                        print(f"Name: {repo.get('name')}")
                        print(f"Full name: {repo.get('full_name')}")
                        print(f"Description: {repo.get('description')}")
                        print(f"Stars: {repo.get('stargazers_count')}")
                        print(f"Forks: {repo.get('forks_count')}")
                        print(f"Language: {repo.get('language')}")
                        print(f"Open issues: {repo.get('open_issues_count')}")
                        print(f"Created at: {repo.get('created_at')}")
                        print(f"Updated at: {repo.get('updated_at')}")
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                print(f"Could not parse repository details: {e}")
        else:
            print(f"Repository search result (type: {type(content).__name__}):")
            print(str(content)[:500])
            
    except Exception as e:
        print(f"Error searching for repository: {e}")

async def sample_config_connection():
    """Sample that demonstrates connecting via a config file"""
    print("\n==== CONFIG CONNECTION SAMPLE ====")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(config, temp_file)
        config_path = temp_file.name
    
    print(f"Created temporary config file at: {config_path}")
    
    try:
        # Create a toolkit to manage multiple clients
        toolkit = MCPToolkit(config_path=config_path)
        
        print("Connecting to all servers in config...")
        try:
            await toolkit.connect()
            print("Connected successfully!")
            
            # Get detailed tools information
            detailed_info = toolkit.detailed_tools_info()
            
            print("\n=== Detailed Tool Information By Server ===")
            for server_name, server_tools in detailed_info.items():
                print(f"\nServer: {server_name}")
                print(f"  Number of tools: {len(server_tools)}")
                
                for i, tool in enumerate(server_tools):
                    print(f"\n  TOOL {i+1}: {tool['name']}")
                    print(f"    Description: {tool['description']}")
                    print(f"    Required parameters: {tool['required_params']}")
                    print("    Parameters:")
                    for param_name, param_schema in tool['parameters'].items():
                        param_type = param_schema.get('type', 'any')
                        param_desc = param_schema.get('description', 'No description')
                        required = "Required" if param_name in tool['required_params'] else "Optional"
                        print(f"      - {param_name} ({param_type}, {required}): {param_desc}")
            
            # Get combined OpenAI tool schemas
            all_schemas = toolkit.get_all_openai_tool_schemas()
            print(f"\n=== Combined OpenAI Tool Schemas ({len(all_schemas)} tools) ===")
            for i, schema in enumerate(all_schemas[:3]):  # Show just first 3 to avoid clutter
                print(f"\nSchema {i+1}: {schema['function']['name']}")
                print(f"  Description: {schema['function']['description']}")
                
            if len(all_schemas) > 3:
                print(f"  ... and {len(all_schemas) - 3} more ...")
                
            # Test repository search for the MCP Python SDK
            if toolkit.servers and toolkit.servers[0].is_connected():
                await search_repo_info(toolkit.servers[0], "modelcontextprotocol", "python-sdk")
                
        except Exception as e:
            print(f"Error connecting to servers: {e}")
        finally:
            print("\nDisconnecting from all servers...")
            await toolkit.disconnect()
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(config_path)
            print(f"Removed temporary config file: {config_path}")
        except Exception as e:
            print(f"Error removing temporary file: {e}")

async def sample_openai_with_mcp():
    """Sample that demonstrates using OpenAILLM with MCP tools to summarize a repository"""
    print("\n==== OPENAI WITH MCP SAMPLE ====")
    
    # Config for GitHub MCP server
    github_config = {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-github"
        ],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "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"
        }
    }
    
    # Initialize the MCP client for GitHub
    mcp_toolkit = MCPToolkit(servers=[MCPClient(**github_config)])
    await mcp_toolkit.connect()
    
    
    try:
        print("Connecting to GitHub MCP server...")
        # Connect to the server
        print("Connected successfully!")
        
        # Get OpenAI-compatible tool schemas
        openai_schemas = mcp_toolkit.get_all_openai_tool_schemas()
        print(f"Retrieved {len(openai_schemas)} tools from GitHub MCP server")
        
        # Set up OpenAILLM with MCP tools
        print("\nInitializing OpenAI LLM with MCP tools...")
        
        # Get OpenAI API key from environment variable or use a test key
        # NOTE: You need to set the OPENAI_API_KEY environment variable with your actual OpenAI API key to run this test
        openai_api_key = "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"
        
        # Create OpenAILLM configuration
        llm_config = OpenAILLMConfig(
            llm_type="OpenAILLM",
            model="gpt-4o-mini",  # Using GPT-4o-mini for tool usage capabilities
            openai_key=openai_api_key,
            temperature=0.7,
            max_tokens=1000,
            tools=openai_schemas,  # Provide the MCP tools to OpenAI
            tool_choice="auto"     # Let the model decide when to use tools
        )
        
        # Create the OpenAILLM instance
        openai_llm = OpenAILLM(config=llm_config)
        
        # Create a helper function to process MCP tool calls
        async def mcp_tool_handler(tool_calls):
            """Process MCP tool calls and return results"""
            results = []
            
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                print(f"\nModel called tool: {tool_name}")
                print(f"Tool arguments: {tool_args}")
                
                try:
                    # Call the MCP tool
                    tool_result = await client.call_tool(tool_name, tool_args)
                    
                    # Format result for display
                    if hasattr(tool_result, 'text'):
                        result_str = tool_result.text
                    else:
                        result_str = str(tool_result)
                    
                    # Truncate if too long
                    if len(result_str) > 1000:
                        print(f"Tool result (truncated): {result_str[:1000]}...")
                    else:
                        print(f"Tool result: {result_str}")
                    
                    # Return the tool result
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "content": result_str
                    })
                except Exception as e:
                    print(f"Error calling tool {tool_name}: {e}")
                    # Return an error message
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "content": f"Error: {str(e)}"
                    })
            
            return results
        
        # Set up a function to handle the conversation with tool calls
        async def generate_with_mcp_tools(messages):
            """Generate text using OpenAILLM with MCP tool support"""
            conversation = messages.copy()  # Working copy of the conversation
            
            cur_iter = 0
            while True:
                cur_iter += 1
                # Use single_generate but with output_response=False to get raw output
                print("\nSending request to OpenAI API...")
                
                # For testing, we'll override the normal parameters to allow tool use
                # We need to add a custom handler to intercept tool calls
                class ToolCallInterceptor:
                    def __init__(self):
                        self.has_tool_calls = False
                        self.assistant_mecssage = None
                    
                    def handle_response(self, response):
                        """Capture the full response object before it's processed"""
                        # Store the complete message with possible tool_calls
                        self.assistant_message = response.choices[0].message
                        # Check if there are tool calls
                        self.has_tool_calls = bool(self.assistant_message.tool_calls)
                        # Return content only when there are no tool calls
                        return self.assistant_message.content
                
                # Create interceptor
                interceptor = ToolCallInterceptor()
                
                try:
                    # Custom function to process the response
                    original_get_completion_output = openai_llm.get_completion_output
                    
                    # Override the get_completion_output method temporarily
                    def custom_get_completion_output(response, output_response=False):
                        return interceptor.handle_response(response)
                    
                    # Apply the monkey patch
                    openai_llm.get_completion_output = custom_get_completion_output
                    
                    # Call single_generate with our conversation
                    result = openai_llm.single_generate(messages=conversation, output_response=False)
                    
                    print("____________________________")
                    print(f"Iteration {cur_iter}: {result}")
                    print(locals())
                    print("____________________________")
                    # Restore original method
                    openai_llm.get_completion_output = original_get_completion_output
                
                finally:
                    # Ensure we restore the original method even if an exception occurs
                    if 'original_get_completion_output' in locals():
                        openai_llm.get_completion_output = original_get_completion_output
                
                # If no tool calls, we're done
                if not interceptor.has_tool_calls:
                    print("\nFinal response (no tool calls):")
                    print(result)
                    return result
                
                # Format the assistant message with tool calls
                assistant_message = interceptor.assistant_message
                formatted_assistant_message = {
                    "role": "assistant",
                    "content": assistant_message.content if assistant_message.content else None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in assistant_message.tool_calls
                    ]
                }
                
                # Add the assistant message to the conversation
                conversation.append(formatted_assistant_message)
                
                # Process the tool calls
                tool_results = await mcp_tool_handler([
                    {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    } for tool_call in assistant_message.tool_calls
                ])
                
                # Add tool results to the conversation
                for result in tool_results:
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"]
                    })
        
        # Prepare the prompt for repository summarization
        repo_to_summarize = "camel-ai/camel"
        system_message = "You are a helpful assistant that can use GitHub search tools to find information about repositories and provide concise summaries."
        prompt = f"""
        Please provide a comprehensive summary of the {repo_to_summarize} repository. 
        Include:
        - The main purpose of the project
        - Key features and functionality
        - Technologies used
        
        Use the available GitHub tools to search for and analyze information about the repository.
        """
        
        print(f"\nSummarizing repository: {repo_to_summarize}")
        print("This may take a moment...")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Generate the summary using OpenAILLM with MCP tools
            result = await generate_with_mcp_tools(messages)
            
            print("\n=== Repository Summary ===")
            print(result)
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error in OpenAI with MCP sample: {e}")
    finally:
        print("\nCleaning up MCP client...")
        await client.cleanup()

async def run_samples():
    """Run all the sample demonstrations"""
    try:
        # Direct connection to a script
        # await sample_direct_connection()
        
        # Connection via config file
        # await sample_config_connection()
        
        # OpenAI LLM with MCP tools
        await sample_openai_with_mcp()
        
    except KeyboardInterrupt:
        print("\nSamples interrupted by user")
    except Exception as e:
        print(f"\nError running samples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Import inspect at the top level since we need it in the direct connection sample
    import inspect
    asyncio.run(run_samples())
