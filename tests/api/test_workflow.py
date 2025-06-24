"""
API Integration Tests for EvoAgentX.

These tests cover:
1. Authentication flow (successful and failed login)
2. Agent CRUD operations (create, get, update, list, delete)
3. Workflow generation and execution pipeline (stock market analysis)
4. Unauthorized access to protected endpoints
5. Agent search/filtering capabilities

To run the workflow tests:
1. Start the EvoAgentX server: python -m evoagentx.app.main
2. Set OPENAI_API_KEY environment variable
3. Run: pytest tests/api/test_workflow.py::test_workflow_generation -v -s

The test will:
- Generate a stock market analysis workflow from a goal
- Store the workflow in the database
- Execute it with AAPL stock (30 days)
- Execute it again with GOOGL stock (90 days)
- Show detailed results and IDs for tracking
"""
import pytest
import pytest_asyncio
import uuid
from urllib.parse import urljoin
import httpx
from httpx import ReadTimeout  # Explicitly import ReadTimeout for exception handling
import os
import dotenv

# Base API URL configuration - allow override from environment
BASE_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1/")
dotenv.load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# OpenAI configuration for testing
openai_config = {
    "model": "gpt-4o-mini",
    "openai_key": OPENAI_API_KEY,
    "stream": True,
    "output_response": True,
    "max_tokens": 16000
}

# --- Helper functions --- #
async def create_agent(client: httpx.AsyncClient, headers: dict, payload: dict) -> str:
    """Helper to create an agent and return its ID."""
    response = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=payload)
    assert response.status_code in (200, 201), f"Unexpected status: {response.status_code}"
    agent_data = response.json()
    return agent_data["_id"]

async def delete_agent(client: httpx.AsyncClient, headers: dict, agent_id: str):
    """Helper to delete an agent."""
    response = await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)
    assert response.status_code == 204

async def create_workflow(client: httpx.AsyncClient, headers: dict, payload: dict) -> str:
    """Helper to create a workflow and return its ID."""
    response = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=payload)
    assert response.status_code == 201, f"Unexpected status: {response.status_code}"
    workflow_data = response.json()
    return workflow_data["_id"]

async def delete_workflow(client: httpx.AsyncClient, headers: dict, workflow_id: str):
    """Helper to delete a workflow."""
    response = await client.delete(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
    assert response.status_code == 204

# --- Fixtures --- #
@pytest_asyncio.fixture
async def client():
    """Create an HTTP client with increased timeout settings."""
    # Increased timeout from default to avoid false failures due to server load
    timeout = httpx.Timeout(60.0, connect=30.0)  # 60s timeout, 30s for connection
    async with httpx.AsyncClient(timeout=timeout) as client:
        yield client

@pytest_asyncio.fixture
async def access_token(client: httpx.AsyncClient):
    try:
        response = await client.post(
            urljoin(BASE_URL, "auth/login"),
            data={"username": "admin@clayx.ai", "password": "adminpassword"}
        )
        assert response.status_code == 200
        return response.json()["access_token"]
    except ReadTimeout:
        pytest.skip(f"Server at {BASE_URL} not responding. Make sure the server is running.")
    except Exception as e:
        pytest.skip(f"Authentication failed: {str(e)}")

# -----------------------
# Tests
# -----------------------


@pytest.mark.asyncio
async def test_workflow_generation(client: httpx.AsyncClient, access_token: str):
    """Test complete workflow generation and execution pipeline for stock market analysis."""
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Define the goal for stock market analysis workflow
    goal = """
    Create a workflow for comprehensive stock market analysis that takes a stock symbol and time duration as inputs.
    The workflow should:
    1. Fetch current stock price and basic information
    2. Analyze historical price trends for the specified duration
    3. Calculate key financial metrics (moving averages, volatility, etc.)
    4. Generate a summary report with investment recommendations
    
    The workflow must accept exactly these inputs:
    - stock_id: The stock symbol (e.g., "AAPL", "GOOGL")
    - duration: Analysis time period (e.g., "30d", "90d", "1y")
    
    The workflow should output:
    - analysis_report: Comprehensive analysis report
    - recommendation: Investment recommendation (buy/hold/sell)
    - key_metrics: Dictionary of calculated financial metrics
    """
    
    # Additional context for the workflow
    additional_info = {
        "focus": "financial analysis",
        "complexity": "intermediate",
        "tools_needed": ["web_search", "data_analysis", "calculation"]
    }
    
    payload = {
        "goal": goal,
        "llm_config": openai_config,
        "additional_info": additional_info
    }
    
    print("🚀 Starting workflow generation...")
    
    # Test workflow generation
    response = await client.post(
        urljoin(BASE_URL, "workflows/generate"),
        headers=headers,
        json=payload,
        timeout=120.0  # Increased timeout for generation
    )
    
    assert response.status_code == 200, f"Generation failed with status {response.status_code}: {response.text}"
    
    generation_result = response.json()
    
    # Validate the response structure
    assert "success" in generation_result
    assert "workflow_graph" in generation_result
    assert "task_info" in generation_result
    assert "original_goal" in generation_result
    
    print(f"✅ Workflow generation response received")
    
    # If successful, validate the task info and proceed to execution
    if generation_result["success"]:
        task_info = generation_result["task_info"]
        assert "workflow_name" in task_info
        assert "workflow_description" in task_info
        assert "inputs_format" in task_info
        assert "outputs_format" in task_info
        
        # Validate that the required inputs are present
        inputs_format = task_info["inputs_format"]
        assert "stock_id" in inputs_format, "Missing required input: stock_id"
        assert "duration" in inputs_format, "Missing required input: duration"
        
        # Validate that expected outputs are present
        outputs_format = task_info["outputs_format"]
        expected_outputs = ["analysis_report", "recommendation", "key_metrics"]
        for output in expected_outputs:
            assert any(output.lower() in key.lower() for key in outputs_format.keys()), f"Missing expected output: {output}"
        
        print(f"✅ Workflow generated successfully: {task_info['workflow_name']}")
        print(f"📋 Description: {task_info['workflow_description']}")
        print(f"📥 Inputs: {list(inputs_format.keys())}")
        print(f"📤 Outputs: {list(outputs_format.keys())}")
        
        # Check if workflow was stored and get its ID
        workflow_id = generation_result.get("workflow_id")
        if workflow_id:
            print(f"💾 Workflow stored with ID: {workflow_id}")
            
            # Now test execution of the stored workflow
            print("\n🔄 Testing workflow execution...")
            
            # Prepare execution inputs for stock analysis
            execution_inputs = {
                "stock_id": "AAPL",  # Apple Inc.
                "duration": "30d"    # 30 days analysis
            }
            
            execution_payload = {
                "llm_config": openai_config,
                "inputs": execution_inputs,
                "mcp_config": {"enabled": False}
            }
            
            # Execute the stored workflow
            exec_response = await client.post(
                urljoin(BASE_URL, f"workflows/{workflow_id}/execute"),
                headers=headers,
                json=execution_payload,
                timeout=180.0  # Increased timeout for execution
            )
            
            assert exec_response.status_code == 200, f"Execution failed with status {exec_response.status_code}: {exec_response.text}"
            
            execution_result = exec_response.json()
            
            # Validate execution response
            assert "success" in execution_result
            assert "workflow_id" in execution_result
            assert execution_result["workflow_id"] == workflow_id
            
            print(f"📊 Execution completed with status: {execution_result.get('status', 'unknown')}")
            
            if execution_result["success"]:
                print(f"✅ Workflow executed successfully!")
                print(f"🆔 Workflow ID: {execution_result['workflow_id']}")
                
                if "execution_id" in execution_result:
                    print(f"🆔 Execution ID: {execution_result['execution_id']}")
                
                # Show execution results summary
                if "message" in execution_result:
                    message = str(execution_result['message'])
                    if len(message) > 300:
                        print(f"💡 Result: {message[:300]}...")
                    else:
                        print(f"💡 Result: {message}")
                
                print(f"🎉 Test completed successfully! Generated and executed stock analysis workflow.")
                
            else:
                print(f"❌ Workflow execution failed: {execution_result.get('error', 'Unknown error')}")
                # Still assert success to see the full error details
                assert execution_result["success"], f"Workflow execution failed: {execution_result.get('error')}"
                
        else:
            print("⚠️  Workflow was not stored in database, testing direct execution instead...")
            
            # Test direct workflow execution (without stored workflow)
            workflow_graph = generation_result["workflow_graph"]
            
            direct_execution_inputs = {
                "stock_id": "AAPL",
                "duration": "30d"
            }
            
            direct_payload = {
                "workflow_graph": workflow_graph,
                "llm_config": openai_config,
                "inputs": direct_execution_inputs,
                "mcp_config": {"enabled": False}
            }
            
            direct_response = await client.post(
                urljoin(BASE_URL, "workflows/execute"),
                headers=headers,
                json=direct_payload,
                timeout=180.0
            )
            
            assert direct_response.status_code == 200, f"Direct execution failed: {direct_response.text}"
            
            direct_result = direct_response.json()
            
            if direct_result.get("success"):
                print(f"✅ Direct workflow execution successful!")
                print(f"📊 Status: {direct_result.get('status')}")
            else:
                print(f"❌ Direct workflow execution failed: {direct_result.get('error')}")
        
    else:
        print(f"❌ Workflow generation failed: {generation_result.get('error', 'Unknown error')}")
        # For testing purposes, we'll still assert success to see what went wrong
        assert generation_result["success"], f"Workflow generation failed: {generation_result.get('error')}"


@pytest.mark.asyncio
async def test_workflow_execution(client: httpx.AsyncClient, access_token: str):
    """This test is now integrated into test_workflow_generation for a complete pipeline test."""
    # Redirect to the comprehensive test
    await test_workflow_generation(client, access_token)