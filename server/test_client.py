import aiohttp
import asyncio
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# PROJECT-BASED WORKFLOW TESTS - STOCK PRICE AND TREND ANALYSIS
# =============================================================================

"""
Complete Stock Price and Trend Analysis Workflow Test Example:

=== INPUT REQUESTS ===

1. PROJECT SETUP INPUT:
{
  "goal": "Create a stock price and trend analysis workflow that can analyze any company's stock performance, including current price metrics, historical trends, and generate comprehensive reports with insights and recommendations",
  "additional_info": {
    "domain": "financial_analysis",
    "output_format": "detailed_report",
    "analysis_type": "comprehensive",
    "data_sources": ["financial_data", "market_trends", "news_sentiment"]
  }
}

2. WORKFLOW GENERATION INPUT:
{
  "project_id": "proj_abc123def456",
  "llm_config": {
    "model": "gpt-4o-mini",
    "openai_key": "your_openai_key_here",
    "stream": true,
    "output_response": true,
    "max_tokens": 8000
  }
}

3. WORKFLOW EXECUTION INPUT:
{
  "project_id": "proj_abc123def456", 
  "inputs": {
    "goal": "Analyze the price and trend for company Apple"
  },
  "llm_config": {
    "model": "gpt-4o-mini",
    "openai_key": "your_openai_key_here",
    "stream": true,
    "output_response": true,
    "max_tokens": 8000
  }
}

=== EXPECTED OUTPUTS ===

1. PROJECT SETUP OUTPUT:
{
  "project_id": "proj_abc123def456",
  "public_url": "https://example.ngrok.io",
  "local_url": "http://localhost:8001",
  "task_info": {
    "workflow_name": "Stock Price and Trend Analysis Workflow",
    "workflow_description": "This workflow analyzes a company's stock performance by evaluating current price metrics, historical trends, and generating comprehensive reports with insights and recommendations based on financial data, market trends, and news sentiment.",
    "workflow_inputs": [
      {"name": "goal", "type": "string", "description": "The user's goal in textual format.", "required": true}
    ],
    "workflow_outputs": [
      {"name": "workflow_output", "type": "string", "description": "A detailed report containing current price metrics, historical trends, insights, and recommendations.", "required": true}
    ],
    "connection_instruction": "# Project Access Instructions\n\n## Project Information\n- **Project ID**: proj_abc123def456\n- **Server URL**: https://example.ngrok.io\n- **Local URL**: http://localhost:8001\n\n## API Endpoints\n\n### 1. Generate Workflow\nCreate a workflow for your project (only needed once per project):\n\n```bash\ncurl -X POST https://example.ngrok.io/workflow/generate \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\n    \"project_id\": \"proj_abc123def456\",\n    \"llm_config\": {\n      \"model\": \"gpt-4o-mini\",\n      \"openai_key\": \"your_openai_api_key\"\n    }\n  }'\n```\n\n### 2. Execute Workflow\nRun the workflow with your specific input:\n\n```bash\ncurl -X POST https://example.ngrok.io/workflow/execute \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\n    \"project_id\": \"proj_abc123def456\",\n    \"inputs\": {\n      \"goal\": \"Analyze the price and trend for company Apple\"\n    },\n    \"llm_config\": {\n      \"model\": \"gpt-4o-mini\",\n      \"openai_key\": \"your_openai_api_key\"\n    }\n  }'\n```\n\n## Quick Start Example\n\n1. **Generate Workflow** (one-time setup):\n   ```bash\n   curl -X POST https://example.ngrok.io/workflow/generate \\\n     -H \"Content-Type: application/json\" \\\n     -d '{\"project_id\": \"proj_abc123def456\"}'\n   ```\n\n2. **Execute Analysis**:\n   ```bash\n   curl -X POST https://example.ngrok.io/workflow/execute \\\n     -H \"Content-Type: application/json\" \\\n     -d '{\n       \"project_id\": \"proj_abc123def456\",\n       \"inputs\": {\"goal\": \"Analyze Tesla stock performance\"}\n     }'\n   ```\n\n## Input Examples\n- \"Analyze the price and trend for company Apple\"\n- \"Provide stock analysis for Microsoft with technical indicators\"  \n- \"Generate investment report for Google including market sentiment\"\n\n## Notes\n- Replace `your_openai_api_key` with your actual OpenAI API key\n- The workflow generates comprehensive stock analysis reports\n- Each execution can analyze different companies by changing the input goal\n- Generated workflows are reusable for multiple executions"
  }
}

2. WORKFLOW GENERATION OUTPUT:
{
  "success": true,
  "project_id": "proj_abc123def456",
  "workflow_graph": {
    "nodes": [...4 workflow nodes...],
    "edges": [...workflow connections...],
    "goal": "Create a stock price and trend analysis workflow...",
    "description": "Generated workflow for stock analysis"
  },
  "workflow_inputs": [
    {"name": "goal", "type": "string", "description": "The user's goal in textual format.", "required": true}
  ],
  "workflow_outputs": [
    {"name": "workflow_output", "type": "string", "description": "A detailed report containing current price metrics, historical trends, insights, and recommendations.", "required": true}
  ],
  "message": "Workflow generated successfully for project",
  "timestamp": "2024-06-18T11:23:55.616000"
}

3. WORKFLOW EXECUTION OUTPUT:
{
  "success": true,
  "project_id": "proj_abc123def456",
  "execution_result": {
    "status": "completed",
    "message": "# Comprehensive Report: AAPL Stock Performance Analysis\n\n### 1. Current Price Metrics\n- **Stock Symbol**: AAPL\n- **Latest Stock Price**: $175.30\n- **Market Capitalization**: $2.8 Trillion\n- **Volume Traded**: 95 Million Shares\n\n### 2. Historical Price Data (Last 5 Years)\n- **Average Price**: $150.45\n- **Peak Price**: $182.50 (November 2021)\n- **Low Price**: $84.80 (March 2020)\n- **Volatility**: Notable fluctuations during earnings releases...\n\n### 3. Key Performance Metrics\n- **1-Year Change**: +15%\n- **5-Year Change**: +20%\n- **Dividend Yield**: 0.55%\n\n### 4. Technical Indicators\n- **50-day Moving Average**: $170.00\n- **200-day Moving Average**: $160.00\n- **RSI**: 65 (nearing overbought)\n- **MACD**: Positive divergence\n\n### 5. Recommendations\n- Strong buy recommendation based on solid fundamentals\n- Consider adding positions on market dips\n- Monitor economic indicators for potential impacts\n\n### Conclusion\nAPPL presents a compelling investment opportunity with consistent performance, positive market sentiment, and sound fundamentals.",
    "workflow_received": true,
    "llm_config_received": true,
    "mcp_config_received": false
  },
  "workflow_info": "=== WORKFLOW EXECUTION INFORMATION ===\nTimestamp: 2024-06-18T11:25:02.789000\nPublic URL: https://example.ngrok.io\nLocal URL: http://localhost:8001\nWorkflow Status: completed\nLLM Configuration: gpt-4o-mini\nMCP Configuration: Disabled\nExecution Message: [Comprehensive Stock Analysis Report]\nWorkflow Received: True\nLLM Config Received: True\nMCP Config Received: False",
  "message": "Workflow executed successfully for project",
  "timestamp": "2024-06-18T11:25:02.789000"
}
"""

def test_health_check():
    """Test basic health check endpoint"""
    print("\n=== Testing Health Check ===")
    
    response = requests.get('http://localhost:8001/health')
    assert response.status_code == 200
    
    data = response.json()
    print("✅ Health check passed:", data)
    return data

def test_project_setup():
    """
    Test project setup for stock price and trend analysis workflow
    
    Curl command:
    ```bash
    curl -X POST http://localhost:8001/project/setup \
      -H "Content-Type: application/json" \
      -d '{
        "goal": "Create a stock price and trend analysis workflow",
        "additional_info": {
          "domain": "financial_analysis",
          "output_format": "detailed_report"
        }
      }'
    ```
    """
    print("\n=== Testing Project Setup - Stock Analysis ===")
    
    project_request = {
        "goal": "Create a stock price and trend analysis workflow that can analyze any company's stock performance, including current price metrics, historical trends, and generate comprehensive reports with insights and recommendations",
        "additional_info": {
            "domain": "financial_analysis",
            "output_format": "detailed_report",
            "analysis_type": "comprehensive",
            "data_sources": ["financial_data", "market_trends", "news_sentiment"]
        }
    }
    
    print(f"🚀 Setting up stock analysis project...")
    print(f"   Goal: {project_request['goal'][:80]}...")
    
    response = requests.post('http://localhost:8001/project/setup', json=project_request)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Project created successfully!")
        print(f"   Project ID: {result['project_id']}")
        print(f"   Public URL: {result['public_url']}")
        print(f"   Local URL: {result['local_url']}")
        print(f"\n📄 Task Info:")
        print(json.dumps(result['task_info'], indent=2))
        
        return result
    else:
        print(f"❌ Project setup failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_project_status(project_id):
    """Test retrieving project status"""
    print(f"\n=== Testing Project Status for {project_id} ===")
    
    response = requests.get(f'http://localhost:8001/project/{project_id}/status')
    
    if response.status_code == 200:
        status = response.json()
        
        print(f"✅ Project status retrieved:")
        print(f"   Status: {status['status']}")
        print(f"   Workflow Generated: {status['workflow_generated']}")
        print(f"   Workflow Executed: {status['workflow_executed']}")
        print(f"   Created: {status['created_at']}")
        print(f"   Last Updated: {status.get('last_updated', 'N/A')}")
        
        return status
    else:
        print(f"❌ Failed to get project status: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_project_workflow_generation(project_id):
    """
    Test workflow generation for stock analysis project
    
    Curl command:
    ```bash
    curl -X POST http://localhost:8001/workflow/generate \
      -H "Content-Type: application/json" \
      -d '{
        "project_id": "proj_abc123def456",
        "llm_config": {
          "model": "gpt-4o-mini",
          "openai_key": "your_key_here"
        }
      }'
    ```
    """
    print(f"\n=== Testing Stock Analysis Workflow Generation for {project_id} ===")
    
    # Default LLM config (can be omitted to use server default)
    llm_config = {
        "model": "gpt-4o-mini",
        "openai_key": OPENAI_API_KEY,
        "stream": True,
        "output_response": True,
        "max_tokens": 8000
    }
    
    generation_request = {
        "project_id": project_id,
        "llm_config": llm_config  # Optional - will use default if omitted
    }
    
    print(f"🚀 Generating stock analysis workflow...")
    print(f"   Using LLM: {llm_config['model']}")
    print(f"   Getting goal and specifications from project data")
    
    response = requests.post('http://localhost:8001/workflow/generate', json=generation_request)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Workflow generated successfully!")
        print(f"   Success: {result['success']}")
        print(f"   Project ID: {result['project_id']}")
        print(f"   Message: {result['message']}")
        print(f"   Timestamp: {result['timestamp']}")
        
        # Show workflow preview - SUMMARY
        workflow_graph = result.get('workflow_graph')
        if isinstance(workflow_graph, dict):
            nodes_count = len(workflow_graph.get('nodes', []))
            edges_count = len(workflow_graph.get('edges', []))
            print(f"   📊 Workflow Structure: {nodes_count} nodes, {edges_count} edges")
            
            # Show workflow node names only (not the full graph)
            nodes = workflow_graph.get('nodes', [])
            if nodes:
                print(f"   📋 Workflow Tasks:")
                for i, node in enumerate(nodes, 1):
                    task_name = node.get('id', f'task_{i}')
                    task_description = node.get('description', 'No description')
                    print(f"      {i}. {task_name}: {task_description[:80]}...")
        else:
            print(f"\n   📄 WORKFLOW:")
            # Show only first 300 characters if it's a string
            workflow_str = str(workflow_graph)
            if len(workflow_str) > 300:
                print(f"   {workflow_str[:300]}...")
            else:
                print(f"   {workflow_str}")
        
        # Show workflow inputs and outputs
        workflow_inputs = result.get('workflow_inputs', [])
        workflow_outputs = result.get('workflow_outputs', [])
        print(f"   📥 Workflow Inputs: {len(workflow_inputs)} inputs")
        for inp in workflow_inputs:
            print(f"      - {inp.get('name', 'unknown')}: {inp.get('description', 'no description')}")
        print(f"   📤 Workflow Outputs: {len(workflow_outputs)} outputs")
        for out in workflow_outputs:
            print(f"      - {out.get('name', 'unknown')}: {out.get('description', 'no description')}")
        
        return result
    else:
        print(f"❌ Workflow generation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_project_workflow_generation_with_default_config(project_id):
    """Test workflow generation using default LLM config (no config provided)"""
    print(f"\n=== Testing Workflow Generation with Default Config for {project_id} ===")
    
    generation_request = {
        "project_id": project_id
        # No llm_config provided - should use default
    }
    
    print(f"🚀 Generating workflow with default config...")
    print(f"   Getting goal and specifications from project data")
    
    response = requests.post('http://localhost:8001/workflow/generate', json=generation_request)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Workflow generated with default config!")
        print(f"   Success: {result['success']}")
        print(f"   Message: {result['message']}")
        
        # Show workflow inputs and outputs for default config test
        workflow_inputs = result.get('workflow_inputs', [])
        workflow_outputs = result.get('workflow_outputs', [])
        print(f"   📥 Generated {len(workflow_inputs)} inputs, 📤 {len(workflow_outputs)} outputs")
        
        return result
    else:
        print(f"❌ Workflow generation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_list_projects():
    """Test listing all projects"""
    print(f"\n=== Testing Project Listing ===")
    
    response = requests.get('http://localhost:8001/projects')
    
    if response.status_code == 200:
        projects = response.json()
        
        print(f"✅ Projects retrieved:")
        print(f"   Total: {projects['total_count']}")
        print(f"   Active: {len(projects['active_projects'])}")
        print(f"   All Projects: {projects['projects']}")
        
        return projects
    else:
        print(f"❌ Failed to list projects: {response.status_code}")
        print(f"   Error: {response.text}")
        return None
    
def test_invalid_project():
    """Test behavior with invalid project ID"""
    print(f"\n=== Testing Invalid Project Handling ===")
    
    invalid_project_id = "proj_nonexistent123"
    
    # Test invalid project status
    response = requests.get(f'http://localhost:8001/project/{invalid_project_id}/status')
    print(f"Status check for invalid project: {response.status_code}")
    
    # Test workflow generation for invalid project
    generation_request = {
        "project_id": invalid_project_id
    }
    
    response = requests.post('http://localhost:8001/workflow/generate', json=generation_request)
    print(f"Workflow generation for invalid project: {response.status_code}")
    
    if response.status_code == 400:
        error = response.json()
        print(f"✅ Proper error handling: {error.get('detail', 'Unknown error')}")
        return True
    else:
        print(f"⚠️  Unexpected response: {response.text}")
        return False

def test_project_workflow_execution(project_id):
    """
    Test workflow execution for Apple stock analysis
    
    Curl command:
    ```bash
    curl -X POST http://localhost:8001/workflow/execute \
      -H "Content-Type: application/json" \
      -d '{
        "project_id": "proj_abc123def456",
        "inputs": {
          "goal": "Analyze the price and trend for company Apple"
        },
        "llm_config": {
          "model": "gpt-4o-mini",
          "openai_key": "your_key_here"
        }
      }'
    ```
    """
    print(f"\n=== Testing Apple Stock Analysis Workflow Execution for {project_id} ===")
    
    # Default LLM config (can be omitted to use server default)
    llm_config = {
        "model": "gpt-4o-mini",
        "openai_key": OPENAI_API_KEY,
        "stream": True,
        "output_response": True,
        "max_tokens": 8000
    }
    
    execution_request = {
        "project_id": project_id,
        "inputs": {
            "goal": "Analyze the price and trend for company Apple"
        },
        "llm_config": llm_config  # Optional - will use default if omitted
    }
    
    print(f"🚀 Executing Apple stock analysis workflow...")
    print(f"   Using LLM: {llm_config['model']}")
    print(f"   Analysis Target: Apple Inc.")
    print(f"   Input keys: {list(execution_request['inputs'].keys())}")
    
    response = requests.post('http://localhost:8001/workflow/execute', json=execution_request)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Apple stock analysis completed successfully!")
        print(f"   Success: {result['success']}")
        print(f"   Project ID: {result['project_id']}")
        print(f"   Message: {result['message']}")
        print(f"   Timestamp: {result['timestamp']}")
        
        # Show execution results - FULL OUTPUT
        execution_result = result.get('execution_result')
        if execution_result:
            print(f"   📊 Execution Status: {execution_result.get('status', 'unknown')}")
            execution_message = execution_result.get('message', 'N/A')
            if isinstance(execution_message, dict):
                print(f"\n   📝 FULL ANALYSIS RESULTS:")
                for key, value in execution_message.items():
                    print(f"      📋 {key}:")
                    print(f"      {str(value)}")
                    print()
            else:
                print(f"\n   📝 FULL EXECUTION MESSAGE:")
                print(f"   {str(execution_message)}")
        
        # Show workflow info - FULL OUTPUT
        workflow_info = result.get('workflow_info')
        if workflow_info:
            print(f"\n   📄 FULL WORKFLOW INFO:")
            print(f"   {workflow_info}")
        
        return result
    else:
        print(f"❌ Apple stock analysis failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Stock Price and Trend Analysis Workflow Tests...")
    
    test_health_check()
    
    project_result = test_project_setup()
    if not project_result:
        print("❌ Project setup failed, stopping test")
        raise Exception("Project setup failed")
    project_id = project_result['project_id']
    
    # test_project_status(project_id)
    
    # workflow_result = test_project_workflow_generation(project_id)
    # if not workflow_result:
    #     print("❌ Workflow generation failed")
    #     return False
    
    # test_project_status(project_id)
    
    test_project_workflow_generation_with_default_config(project_id)
    
    test_project_workflow_execution(project_id)

    # test_list_projects()
    
    # test_invalid_project()
    
    print("\n🏁 Stock analysis workflow test execution completed.") 