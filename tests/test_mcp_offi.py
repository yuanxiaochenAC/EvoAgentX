import asyncio
import os
import sys
import json
import tempfile
import shutil
import http.server
import socketserver
import threading
import time
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

# Sample config for PDF reader MCP server
config = {
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": [
                "@modelcontextprotocol/server-github"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "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"
            }
        }
    }
}
config_path = "tests/mcp.config"

async def sample_openai_with_mcp():
    """Sample that demonstrates using OpenAILLM with MCP tools to summarize a repository"""
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
    #     json.dump(config, temp_file)
    #     config_path = temp_file.name
    toolkit = MCPToolkit(config_path=config_path)
    await toolkit.connect()
        
    
    print(f"Created temporary config file at: {config_path}")
    
    try:
        # Get OpenAI-compatible tool schemas
        openai_schemas = toolkit.get_all_openai_tool_schemas()
        print(f"Retrieved {len(openai_schemas)} tools from GitHub MCP server")
        
        # Set up OpenAILLM with MCP tools
        print("\nInitializing OpenAI LLM with MCP tools...")
        
        # Get OpenAI API key from environment variable or use a test key
        # NOTE: You need to set the OPENAI_API_KEY environment variable with your actual OpenAI API key to run this test
        openai_api_key = "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"
        
        # Get OpenAI-compatible tool schemas instead of callable functions
        tools = toolkit.get_tools()
        sample_para = {"query": "camel"}
        print("__________________________")
        print("Tools")
        print(await tools[1][0](**sample_para))
        print("__________________________")
        
        # Create OpenAILLM configuration
        llm_config = OpenAILLMConfig(
            llm_type="OpenAILLM",
            model="gpt-4o-mini",  # Using GPT-4o-mini for tool usage capabilities
            openai_key=openai_api_key,
            temperature=0.7,
            max_tokens=1000,
            tools=tools,  # Use OpenAI-compatible tool schemas
            tool_choice="auto"     # Let the model decide when to use tools
        )
        
        # Create the OpenAILLM instance
        openai_llm = OpenAILLM(config=llm_config)
        
        
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
            print(messages)
            result = openai_llm.single_generate(messages, output_response=False)
            messages.append({
                "role": "assistant",
                    "content": None,
                    "tool_calls": []
            })
            # result = openai_llm.single_generate(messages, output_response=False)
            print("____________________________")
            print(messages)
            print(result)
            print("____________________________")
            
            print(openai_llm.get_completion_output(result, output_response=False))
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
        await toolkit.disconnect()

async def run_samples():
    """Run all the sample demonstrations"""
    try:
        await sample_openai_with_mcp()
        
    except KeyboardInterrupt:
        print("\nSamples interrupted by user")
    except Exception as e:
        print(f"\nError running samples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_samples())
