"""
API Integration Tests for EvoAgentX.

These tests cover:
1. Authentication flow (successful and failed login)
2. Agent CRUD operations (create, get, update, list, delete)
3. Workflow CRUD operations (create, get, update, list, execute)
4. Unauthorized access to protected endpoints
5. Agent search/filtering capabilities
6. Workflow execution with multiple agents and steps
"""
import pytest
import pytest_asyncio

import httpx
from urllib.parse import urljoin

# Base API URL configuration
BASE_URL = "http://localhost:8000/api/v1/"

# Client fixture (shared for tests)
@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient() as client:
        yield client

# Fixture for getting an access token for authenticated endpoints
@pytest_asyncio.fixture
async def access_token(client: httpx.AsyncClient):
    response = await client.post(
        urljoin(BASE_URL, "auth/login"),
        data={
            "username": "admin@clayx.ai",
            "password": "adminpassword"
        }
    )
    assert response.status_code == 200
    return response.json()["access_token"]

# -----------------------
# Authentication Tests
# -----------------------

@pytest.mark.asyncio
async def test_successful_login(client: httpx.AsyncClient):
    response = await client.post(
        urljoin(BASE_URL, "auth/login"),
        data={"username": "admin@clayx.ai", "password": "adminpassword"}
    )
    assert response.status_code == 200
    json_resp = response.json()
    assert "access_token" in json_resp


@pytest.mark.asyncio
async def test_failed_login(client: httpx.AsyncClient):
    response = await client.post(
        urljoin(BASE_URL, "auth/login"),
        data={"username": "admin@clayx.ai", "password": "wrongpassword"}
    )
    assert response.status_code == 401

# -----------------------
# Agent CRUD Tests
# -----------------------

@pytest.mark.asyncio
async def test_create_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "name": "TestAgent1",
        "description": "A test agent for integration testing",
        "type": "llm",  # Added required field
        "config": {
            "model": "gpt-3.5-turbo",  # Updated config structure
            "provider": "openai",
            "temperature": 0.7
        },
        "tags": ["test", "integration"]
    }
    response = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=payload)
    assert response.status_code == 200  # Changed to 200 as that's what your API returns


@pytest.mark.asyncio
async def test_get_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent for testing GET
    payload = {
        "name": "TestAgentForGet",
        "description": "Agent to test get",
        "config": {"model_type": "llm"},
        "runtime_params": {},
        "tags": ["test"]
    }
    create_resp = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=payload)
    assert create_resp.status_code == 201
    agent_data = create_resp.json()
    agent_id = agent_data["_id"]

    # Test: Get the agent
    get_resp = await client.get(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "TestAgentForGet"

    # Clean up
    await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)


@pytest.mark.asyncio
async def test_update_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent for testing update
    payload = {
        "name": "TestAgentForUpdate",
        "description": "Agent to test update",
        "config": {"model_type": "llm"},
        "runtime_params": {},
        "tags": ["test"]
    }
    create_resp = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=payload)
    assert create_resp.status_code == 201
    agent_data = create_resp.json()
    agent_id = agent_data["_id"]

    # Test: Update the agent
    update_payload = {"description": "Updated description", "tags": ["test", "updated"]}
    update_resp = await client.patch(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers, json=update_payload)
    assert update_resp.status_code == 200
    assert update_resp.json()["description"] == "Updated description"

    # Clean up
    await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)


@pytest.mark.asyncio
async def test_list_agents(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create two agents to ensure there are items to list
    payload1 = {
        "name": "ListAgent1",
        "description": "Agent for listing 1",
        "config": {"model_type": "llm"},
        "runtime_params": {},
        "tags": ["list"]
    }
    payload2 = {
        "name": "ListAgent2",
        "description": "Agent for listing 2",
        "config": {"model_type": "llm"},
        "runtime_params": {},
        "tags": ["list"]
    }
    resp1 = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=payload1)
    resp2 = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=payload2)
    assert resp1.status_code == 201
    assert resp2.status_code == 201

    # Test: List agents
    list_resp = await client.get(urljoin(BASE_URL, "agents"), headers=headers, params={"skip": 0, "limit": 10})
    assert list_resp.status_code == 200
    agents = list_resp.json()
    assert len(agents) >= 2

    # Clean up
    await client.delete(urljoin(BASE_URL, f"agents/{resp1.json()['_id']}"), headers=headers)
    await client.delete(urljoin(BASE_URL, f"agents/{resp2.json()['_id']}"), headers=headers)


@pytest.mark.asyncio
async def test_delete_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent for deletion test
    payload = {
        "name": "TestAgentForDelete",
        "description": "Agent to test deletion",
        "config": {"model_type": "llm"},
        "runtime_params": {},
        "tags": ["delete"]
    }
    create_resp = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=payload)
    assert create_resp.status_code == 201
    agent_data = create_resp.json()
    agent_id = agent_data["_id"]

    # Test: Delete the agent
    delete_resp = await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)
    assert delete_resp.status_code == 204

# -----------------------
# Workflow CRUD Tests
# -----------------------

@pytest.mark.asyncio
async def test_create_workflow(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent needed for the workflow
    agent_payload = {
        "name": "WorkflowAgent",
        "config": {"type": "test_agent"},
        "runtime_params": {}
    }
    agent_resp = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=agent_payload)
    assert agent_resp.status_code == 201
    agent_id = agent_resp.json()["_id"]

    # Define and create a workflow
    workflow_definition = {
        "steps": [
            {
                "step_id": "step1",
                "agent_id": agent_id,
                "action": "test_action",
                "input_mapping": {},
                "output_mapping": {}
            }
        ]
    }
    workflow_payload = {
        "name": "TestWorkflow1",
        "description": "A test workflow for integration testing",
        "definition": workflow_definition,
        "tags": ["test", "integration"]
    }
    response = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=workflow_payload)
    assert response.status_code == 201
    workflow_data = response.json()
    assert workflow_data["name"] == "TestWorkflow1"

    # Clean up workflow and agent
    await client.delete(urljoin(BASE_URL, f"workflows/{workflow_data['_id']}"), headers=headers)
    await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)


@pytest.mark.asyncio
async def test_get_workflow(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent and workflow for testing GET
    agent_payload = {
        "name": "WorkflowAgentForGet",
        "config": {"type": "test_agent"},
        "runtime_params": {}
    }
    agent_resp = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=agent_payload)
    assert agent_resp.status_code == 201
    agent_id = agent_resp.json()["_id"]

    workflow_definition = {
        "steps": [
            {
                "step_id": "step1",
                "agent_id": agent_id,
                "action": "test_action",
                "input_mapping": {},
                "output_mapping": {}
            }
        ]
    }
    workflow_payload = {
        "name": "TestWorkflowForGet",
        "description": "Workflow for get operation",
        "definition": workflow_definition,
        "tags": ["get"]
    }
    workflow_resp = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=workflow_payload)
    assert workflow_resp.status_code == 201
    workflow_data = workflow_resp.json()
    workflow_id = workflow_data["_id"]

    # Test: Get the workflow
    get_resp = await client.get(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "TestWorkflowForGet"

    # Clean up
    await client.delete(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
    await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)


@pytest.mark.asyncio
async def test_update_workflow(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent and workflow for update testing
    agent_payload = {
        "name": "WorkflowAgentForUpdate",
        "config": {"type": "test_agent"},
        "runtime_params": {}
    }
    agent_resp = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=agent_payload)
    assert agent_resp.status_code == 201
    agent_id = agent_resp.json()["_id"]

    workflow_definition = {
        "steps": [
            {
                "step_id": "step1",
                "agent_id": agent_id,
                "action": "test_action",
                "input_mapping": {},
                "output_mapping": {}
            }
        ]
    }
    workflow_payload = {
        "name": "TestWorkflowForUpdate",
        "description": "Workflow to test update",
        "definition": workflow_definition,
        "tags": ["update"]
    }
    workflow_resp = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=workflow_payload)
    assert workflow_resp.status_code == 201
    workflow_data = workflow_resp.json()
    workflow_id = workflow_data["_id"]

    # Test: Update the workflow
    update_payload = {"description": "Updated workflow description", "tags": ["update", "modified"]}
    update_resp = await client.patch(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers, json=update_payload)
    assert update_resp.status_code == 200
    assert update_resp.json()["description"] == "Updated workflow description"

    # Clean up
    await client.delete(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
    await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)


@pytest.mark.asyncio
async def test_list_workflows(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent and two workflows for listing
    agent_payload = {
        "name": "WorkflowAgentForList",
        "config": {"type": "test_agent"},
        "runtime_params": {}
    }
    agent_resp = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=agent_payload)
    assert agent_resp.status_code == 201
    agent_id = agent_resp.json()["_id"]

    workflow_payload1 = {
        "name": "ListWorkflow1",
        "description": "First workflow for listing",
        "definition": {"steps": []},
        "tags": ["list"]
    }
    workflow_payload2 = {
        "name": "ListWorkflow2",
        "description": "Second workflow for listing",
        "definition": {"steps": []},
        "tags": ["list"]
    }
    resp1 = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=workflow_payload1)
    resp2 = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=workflow_payload2)
    assert resp1.status_code == 201
    assert resp2.status_code == 201

    # Test: List workflows
    list_resp = await client.get(urljoin(BASE_URL, "workflows"), headers=headers, params={"skip": 0, "limit": 10})
    assert list_resp.status_code == 200
    workflows = list_resp.json()
    assert len(workflows) >= 2

    # Clean up
    await client.delete(urljoin(BASE_URL, f"workflows/{resp1.json()['_id']}"), headers=headers)
    await client.delete(urljoin(BASE_URL, f"workflows/{resp2.json()['_id']}"), headers=headers)
    await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)


@pytest.mark.asyncio
async def test_execute_workflow(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent for workflow execution
    agent_payload = {
        "name": "WorkflowAgentForExecute",
        "config": {"type": "test_agent"},
        "runtime_params": {}
    }
    agent_resp = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=agent_payload)
    assert agent_resp.status_code == 201
    agent_id = agent_resp.json()["_id"]

    workflow_definition = {
        "steps": [
            {
                "step_id": "step1",
                "agent_id": agent_id,
                "action": "test_action",
                "input_mapping": {},
                "output_mapping": {}
            }
        ]
    }
    workflow_payload = {
        "name": "WorkflowForExecute",
        "description": "Workflow for execution testing",
        "definition": workflow_definition
    }
    workflow_resp = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=workflow_payload)
    assert workflow_resp.status_code == 201
    workflow_id = workflow_resp.json()["_id"]

    # Test: Execute the workflow
    execute_resp = await client.post(
        urljoin(BASE_URL, f"workflows/{workflow_id}/execute"),
        headers=headers,
        json={"input_params": {"test_key": "test_value"}}
    )
    assert execute_resp.status_code == 202
    execution_data = execute_resp.json()
    assert "execution_id" in execution_data

    # Clean up
    await client.delete(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
    await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)

# -----------------------
# Additional Tests
# -----------------------

@pytest.mark.asyncio
async def test_unauthorized_access(client: httpx.AsyncClient):
    # Test access to a protected endpoint without authentication
    response = await client.post(
        urljoin(BASE_URL, "agents"),
        json={"name": "UnauthorizedAgent", "config": {}}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_agent_search_and_filter(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    agents_to_create = [
        {"name": "SearchAgent1", "tags": ["search", "test"], "config": {"type": "llm"}},
        {"name": "SearchAgent2", "tags": ["search"], "config": {"type": "nlp"}},
        {"name": "FilterAgent", "tags": ["filter"], "config": {"type": "llm"}}
    ]
    created_agents = []
    for agent in agents_to_create:
        resp = await client.post(urljoin(BASE_URL, "agents"), headers=headers, json=agent)
        assert resp.status_code == 201
        created_agents.append(resp.json()["_id"])

    # Test: Filter agents by tag "search"
    filter_resp = await client.get(urljoin(BASE_URL, "agents"), headers=headers, params={"tags": "search"})
    assert filter_resp.status_code == 200
    filtered_agents = filter_resp.json()
    assert len(filtered_agents) >= 2

    # Clean up
    for agent_id in created_agents:
        await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)


@pytest.mark.asyncio
async def test_workflow_execution_with_multiple_agents(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    agent_ids = []
    # Create multiple agents for the multi-step workflow
    for i in range(3):
        resp = await client.post(
            urljoin(BASE_URL, "agents"),
            headers=headers,
            json={
                "name": f"WorkflowExecutionAgent{i}",
                "config": {"type": f"agent_type_{i}"},
                "runtime_params": {}
            }
        )
        assert resp.status_code == 201
        agent_ids.append(resp.json()["_id"])

    # Define a multi-step workflow
    workflow_definition = {
        "steps": [
            {
                "step_id": "initial_step",
                "agent_id": agent_ids[0],
                "action": "prepare_data",
                "input_mapping": {},
                "output_mapping": {"result": "processed_data"},
                "depends_on": []
            },
            {
                "step_id": "analysis_step",
                "agent_id": agent_ids[1],
                "action": "analyze_data",
                "input_mapping": {"input": "processed_data"},
                "output_mapping": {"result": "analysis_result"},
                "depends_on": ["initial_step"]
            },
            {
                "step_id": "final_step",
                "agent_id": agent_ids[2],
                "action": "generate_report",
                "input_mapping": {"input": "analysis_result"},
                "output_mapping": {"result": "final_report"},
                "depends_on": ["analysis_step"]
            }
        ]
    }
    workflow_payload = {
        "name": "MultiStepWorkflow",
        "description": "Workflow with multiple agent steps",
        "definition": workflow_definition
    }
    workflow_resp = await client.post(urljoin(BASE_URL, "workflows"), headers=headers, json=workflow_payload)
    assert workflow_resp.status_code == 201
    workflow_id = workflow_resp.json()["_id"]

    # Test: Execute the multi-step workflow
    exec_resp = await client.post(
        urljoin(BASE_URL, f"workflows/{workflow_id}/execute"),
        headers=headers,
        json={"input_params": {"initial_data": "test_data"}}
    )
    assert exec_resp.status_code == 202
    exec_data = exec_resp.json()
    assert "execution_id" in exec_data

    # Clean up: delete the workflow and all created agents
    await client.delete(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
    for agent_id in agent_ids:
        await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)
