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
        "command_or_url": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-github"
        ],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "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"
        }
    }
    
    # Initialize the MCP client for GitHub
    client = MCPClient(**github_config)
    
    try:
        print("Connecting to GitHub MCP server...")
        # Connect to the server
        await client.connect()
        print("Connected successfully!")
        
        # Get OpenAI-compatible tool schemas
        openai_schemas = client.get_openai_tool_schemas()
        print(f"Retrieved {len(openai_schemas)} tools from GitHub MCP server")
        
        # Set up OpenAILLM with MCP tools
        print("\nInitializing OpenAI LLM with MCP tools...")
        
        # Get OpenAI API key from environment variable or use a test key
        # NOTE: You need to set the OPENAI_API_KEY environment variable with your actual OpenAI API key to run this test
        openai_api_key = "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"
        
        # Create OpenAILLM configuration
        llm_config = OpenAILLMConfig(
            llm_type="OpenAILLM",
            model="gpt-4o-mini",  # Using GPT-4o as it has good tool usage capabilities
            openai_key=openai_api_key,
            temperature=0.7,
            max_tokens=1000,
            tools=openai_schemas,  # Provide the MCP tools to OpenAI
            tool_choice="auto"     # Let the model decide when to use tools
        )
        
        # Create the OpenAILLM instance
        openai_llm = OpenAILLM(config=llm_config)
        
        # Prepare a direct client for handling our own tool calls
        openai_client = OpenAI(api_key=openai_api_key)
        
        # Function to handle tool calls
        async def process_with_tools(messages):
            print("\nSending request to OpenAI API...")
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=openai_schemas,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )
            
            # Check if model wanted to call a tool
            message = response.choices[0].message
            
            if not message.tool_calls:
                # No tool calls, just return the content
                return message.content
            
            # Add the assistant message with tool_calls to our conversation
            messages.append({
                "role": "assistant",
                "content": message.content if message.content else None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    } for tool_call in message.tool_calls
                ]
            })
            
            # Process tool calls
            for tool_call in message.tool_calls:
                # Extract tool info
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"\nModel called tool: {function_name}")
                print(f"Tool arguments: {function_args}")
                
                # Call the MCP tool
                try:
                    tool_result = await client.call_tool(function_name, function_args)
                    
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
                    
                    # Add the tool result as a separate message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_str
                    })
                except Exception as e:
                    print(f"Error calling tool {function_name}: {e}")
                    # Add an error message as the tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"Error: {str(e)}"
                    })
            
            # After processing all tool calls, ask the model to continue
            print("\nSending follow-up request with tool results...")
            final_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=openai_schemas,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )
            
            final_message = final_response.choices[0].message
            
            # Check if we need to iterate again (more tool calls)
            if final_message.tool_calls:
                # Add the message with tool calls to our conversation
                messages.append({
                    "role": "assistant",
                    "content": final_message.content if final_message.content else None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in final_message.tool_calls
                    ]
                })
                # Recursively process more tool calls
                return await process_with_tools(messages)
            else:
                # No more tool calls, add the final assistant message
                messages.append({
                    "role": "assistant",
                    "content": final_message.content
                })
                # Return the final content
                return final_message.content
        
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
            # Process the request with tools
            result = await process_with_tools(messages)
            
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
