import aiohttp
import asyncio
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# PROJECT-BASED WORKFLOW TESTS
# =============================================================================

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
    Test project setup - the main entry point for the new system
    
    Curl command:
    ```bash
    curl -X POST http://localhost:8001/project/setup \
      -H "Content-Type: application/json" \
      -d '{
        "goal": "Create a comprehensive market analysis workflow",
        "additional_info": {
          "industry": "technology",
          "timeframe": "Q4 2024"
        }
      }'
    ```
    """
    print("\n=== Testing Project Setup ===")
    
    project_request = {
        "goal": "Create a comprehensive market analysis workflow that analyzes current trends, competitor data, and generates actionable insights for the technology sector",
        "additional_info": {
            "industry": "technology",
            "timeframe": "Q4 2024",
            "target_audience": "executives",
            "data_sources": ["financial_reports", "market_research", "news_articles"]
        }
    }
    
    print(f"🚀 Setting up new project...")
    print(f"   Goal: {project_request['goal'][:80]}...")
    
    response = requests.post('http://localhost:8001/project/setup', json=project_request)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Project created successfully!")
        print(f"   Project ID: {result['project_id']}")
        print(f"   Public URL: {result['public_url']}")
        print(f"   Local URL: {result['local_url']}")
        print(f"\n📄 Task Info Preview:")
        print(result['task_info'][:500] + "..." if len(result['task_info']) > 500 else result['task_info'])
        
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
    Test the new project-based workflow generation
    
    Curl command:
    ```bash
    curl -X POST http://localhost:8001/workflow/generate \
      -H "Content-Type: application/json" \
      -d '{
        "project_id": "proj_abc123def456",
        "inputs": "Create a comprehensive market analysis workflow...",
        "llm_config": {
          "model": "gpt-4o-mini",
          "openai_key": "your_key_here"
        }
      }'
    ```
    """
    print(f"\n=== Testing Project Workflow Generation for {project_id} ===")
    
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
        "inputs": """Create a comprehensive market analysis workflow that includes the following steps:
        1. Data Collection: Gather market data from multiple sources including financial reports, news articles, and industry research
        2. Trend Analysis: Analyze market trends and identify key patterns
        3. Competitor Analysis: Research and analyze main competitors in the technology sector
        4. Risk Assessment: Identify potential risks and opportunities
        5. Report Generation: Generate a detailed executive summary with actionable insights
        6. Visualization: Create charts and graphs to support the findings
        
        The workflow should be designed for Q4 2024 technology sector analysis and target executive-level decision makers.""",
        "llm_config": llm_config  # Optional - will use default if omitted
    }
    
    print(f"🚀 Generating workflow for project...")
    print(f"   Using LLM: {llm_config['model']}")
    print(f"   Input length: {len(generation_request['inputs'])} characters")
    
    response = requests.post('http://localhost:8001/workflow/generate', json=generation_request)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Workflow generated successfully!")
        print(f"   Success: {result['success']}")
        print(f"   Project ID: {result['project_id']}")
        print(f"   Message: {result['message']}")
        print(f"   Timestamp: {result['timestamp']}")
        
        # Show workflow preview
        workflow_graph = result.get('workflow_graph')
        if isinstance(workflow_graph, dict):
            nodes_count = len(workflow_graph.get('nodes', []))
            edges_count = len(workflow_graph.get('edges', []))
            print(f"   📊 Workflow Structure: {nodes_count} nodes, {edges_count} edges")
        else:
            print(f"   📄 Workflow: {str(workflow_graph)[:200]}...")
        
        return result
    else:
        print(f"❌ Workflow generation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_project_workflow_generation_with_default_config(project_id):
    """Test workflow generation using default LLM config (no config provided)"""
    print(f"\n=== Testing Workflow Generation with Default Config for {project_id} ===")
    
    generation_request = {
        "project_id": project_id,
        "inputs": "Create a simple data processing workflow that reads CSV files, processes the data, and generates a summary report with basic statistics and visualizations."
        # No llm_config provided - should use default
    }
    
    print(f"🚀 Generating workflow with default config...")
    print(f"   Input: {generation_request['inputs'][:80]}...")
    
    response = requests.post('http://localhost:8001/workflow/generate', json=generation_request)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Workflow generated with default config!")
        print(f"   Success: {result['success']}")
        print(f"   Message: {result['message']}")
        
        print(result)
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
        "project_id": invalid_project_id,
        "inputs": "Test input for invalid project"
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
    Test the new project-based workflow execution
    
    Curl command:
    ```bash
    curl -X POST http://localhost:8001/workflow/execute \
      -H "Content-Type: application/json" \
      -d '{
        "project_id": "proj_abc123def456",
        "inputs": {
          "input": "Market analysis data",
          "timeframe": "Q4 2024"
        },
        "llm_config": {
          "model": "gpt-4o-mini",
          "openai_key": "your_key_here"
        }
      }'
    ```
    """
    print(f"\n=== Testing Project Workflow Execution for {project_id} ===")
    
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
            "input": "Please analyze the technology market trends for Q4 2024, focusing on AI and machine learning sectors. Include competitor analysis, market size, growth projections, and investment opportunities.",
            "timeframe": "Q4 2024",
            "focus_sectors": ["AI", "machine_learning", "cloud_computing"],
            "analysis_depth": "comprehensive",
            "target_audience": "executives"
        },
        "llm_config": llm_config  # Optional - will use default if omitted
    }
    
    print(f"🚀 Executing workflow for project...")
    print(f"   Using LLM: {llm_config['model']}")
    print(f"   Input keys: {list(execution_request['inputs'].keys())}")
    
    response = requests.post('http://localhost:8001/workflow/execute', json=execution_request)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Workflow executed successfully!")
        print(f"   Success: {result['success']}")
        print(f"   Project ID: {result['project_id']}")
        print(f"   Message: {result['message']}")
        print(f"   Timestamp: {result['timestamp']}")
        
        # Show execution results preview
        execution_result = result.get('execution_result')
        if execution_result:
            print(f"   📊 Execution Status: {execution_result.get('status', 'unknown')}")
            print(f"   📝 Execution Message: {execution_result.get('message', 'N/A')[:100]}...")
        
        # Show workflow info preview
        workflow_info = result.get('workflow_info')
        if workflow_info:
            print(f"   📄 Workflow Info: {workflow_info[:200]}...")
        
        return result
    else:
        print(f"❌ Workflow execution failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Project-Based Workflow System Tests...")
    
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
    
    print("\n🏁 Test execution completed.") 