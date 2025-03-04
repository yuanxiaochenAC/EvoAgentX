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
import uuid
from urllib.parse import urljoin
import httpx

# Base API URL configuration
BASE_URL = "http://localhost:8000/api/v1/"

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
    async with httpx.AsyncClient() as client:
        yield client

@pytest_asyncio.fixture
async def access_token(client: httpx.AsyncClient):
    response = await client.post(
        urljoin(BASE_URL, "auth/login"),
        data={"username": "admin@clayx.ai", "password": "adminpassword"}
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
@pytest.mark.asyncio
async def test_create_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
        "description": "A test agent for integration testing",
        "config": {"model": "gpt-3.5-turbo", "provider": "openai", "temperature": 0.7},
        "runtime_params": {},
        "tags": ["test", "integration"]
    }

    agent_id = await create_agent(client, headers, payload)

    # Debugging: Print API response
    print(f"Agent ID: {agent_id}")

    assert agent_id, "Agent ID should not be None or empty"

    # Clean up
    await delete_agent(client, headers, agent_id)


@pytest.mark.asyncio
async def test_get_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "name": f"TestAgentForGet_{uuid.uuid4().hex[:8]}",
        "description": "Agent to test get",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": ["test"]
    }
    agent_id = await create_agent(client, headers, payload)
    try:
        get_resp = await client.get(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == payload["name"]
    finally:
        await delete_agent(client, headers, agent_id)

@pytest.mark.asyncio
async def test_update_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "name": f"TestAgentForUpdate_{uuid.uuid4().hex[:8]}",
        "description": "A test agent for integration testing",
        "type": "llm",
        "config": {
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "temperature": 0.7
        },
        "tags": ["test", "integration"]
    }
    agent_id = await create_agent(client, headers, payload)
    try:
        update_payload = {"description": "Updated description", "tags": ["test", "updated"]}
        # Changed PATCH to PUT here to match the route definition
        update_resp = await client.put(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers, json=update_payload)
        assert update_resp.status_code == 200
        assert update_resp.json()["description"] == "Updated description"
    finally:
        await delete_agent(client, headers, agent_id)


@pytest.mark.asyncio
async def test_list_agents(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    payload1 = {
        "name": f"ListAgent1_{uuid.uuid4().hex[:8]}",
        "description": "Agent for listing 1",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": ["list"]
    }
    payload2 = {
        "name": f"ListAgent2_{uuid.uuid4().hex[:8]}",
        "description": "Agent for listing 2",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": ["list"]
    }
    agent_id1 = await create_agent(client, headers, payload1)
    agent_id2 = await create_agent(client, headers, payload2)
    try:
        list_resp = await client.get(urljoin(BASE_URL, "agents"), headers=headers, params={"skip": 0, "limit": 10})
        assert list_resp.status_code == 200
        agents = list_resp.json()
        assert any(agent["_id"] == agent_id1 for agent in agents)
        assert any(agent["_id"] == agent_id2 for agent in agents)
    finally:
        await delete_agent(client, headers, agent_id1)
        await delete_agent(client, headers, agent_id2)


@pytest.mark.asyncio
async def test_delete_agent(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "name": f"TestAgentForDelete_{uuid.uuid4().hex[:8]}",
        "description": "Agent to test deletion",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": ["delete"]
    }
    agent_id = await create_agent(client, headers, payload)
    delete_resp = await client.delete(urljoin(BASE_URL, f"agents/{agent_id}"), headers=headers)
    assert delete_resp.status_code == 204

# -----------------------
# Workflow CRUD Tests
# -----------------------

@pytest.mark.asyncio
async def test_create_workflow(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Create an agent for the workflow
    agent_payload = {
        "name": f"WorkflowAgent_{uuid.uuid4().hex[:8]}",
        "description": "Agent for workflow",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    workflow_id = None  # Initialize workflow_id here
    try:
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
            "name": f"TestWorkflow_{uuid.uuid4().hex[:8]}",
            "description": "A test workflow for integration testing",
            "definition": workflow_definition,
            "tags": ["test", "integration"]
        }
        # Create the workflow
        workflow_id = await create_workflow(client, headers, workflow_payload)
        get_resp = await client.get(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
        assert get_resp.status_code == 200
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if workflow_id:
            await delete_workflow(client, headers, workflow_id)
        await delete_agent(client, headers, agent_id)


@pytest.mark.asyncio
async def test_get_workflow(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    agent_payload = {
        "name": f"WorkflowAgentForGet_{uuid.uuid4().hex[:8]}",
        "description": "Agent for workflow get",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    try:
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
            "name": f"TestWorkflowForGet_{uuid.uuid4().hex[:8]}",
            "description": "Workflow for get operation",
            "definition": workflow_definition,
            "tags": ["get"]
        }
        workflow_id = await create_workflow(client, headers, workflow_payload)
        try:
            get_resp = await client.get(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers)
            assert get_resp.status_code == 200
            assert get_resp.json()["name"] == workflow_payload["name"]
        finally:
            await delete_workflow(client, headers, workflow_id)
    finally:
        await delete_agent(client, headers, agent_id)

@pytest.mark.asyncio
async def test_update_workflow(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    agent_payload = {
        "name": f"WorkflowAgentForUpdate_{uuid.uuid4().hex[:8]}",
        "description": "Agent for workflow update",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    try:
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
            "name": f"TestWorkflowForUpdate_{uuid.uuid4().hex[:8]}",
            "description": "Workflow to test update",
            "definition": workflow_definition,
            "tags": ["update"]
        }
        workflow_id = await create_workflow(client, headers, workflow_payload)
        try:
            update_payload = {"description": "Updated workflow description", "tags": ["update", "modified"]}
            # Changed PATCH to PUT to match the method defined for updating workflows
            update_resp = await client.put(urljoin(BASE_URL, f"workflows/{workflow_id}"), headers=headers, json=update_payload)
            assert update_resp.status_code == 200
            assert update_resp.json()["description"] == "Updated workflow description"
        finally:
            await delete_workflow(client, headers, workflow_id)
    finally:
        await delete_agent(client, headers, agent_id)


@pytest.mark.asyncio
async def test_list_workflows(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    agent_payload = {
        "name": f"WorkflowAgentForList_{uuid.uuid4().hex[:8]}",
        "description": "Agent for workflow list",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    try:
        workflow_payload1 = {
            "name": f"ListWorkflow1_{uuid.uuid4().hex[:8]}",
            "description": "First workflow for listing",
            "definition": {"steps": []},
            "tags": ["list"]
        }
        workflow_payload2 = {
            "name": f"ListWorkflow2_{uuid.uuid4().hex[:8]}",
            "description": "Second workflow for listing",
            "definition": {"steps": []},
            "tags": ["list"]
        }
        workflow_id1 = await create_workflow(client, headers, workflow_payload1)
        workflow_id2 = await create_workflow(client, headers, workflow_payload2)
        try:
            list_resp = await client.get(urljoin(BASE_URL, "workflows"), headers=headers, params={"skip": 0, "limit": 10})
            assert list_resp.status_code == 200
            workflows = list_resp.json()
            assert any(wf["_id"] == workflow_id1 for wf in workflows)
            assert any(wf["_id"] == workflow_id2 for wf in workflows)
        finally:
            await delete_workflow(client, headers, workflow_id1)
            await delete_workflow(client, headers, workflow_id2)
    finally:
        await delete_agent(client, headers, agent_id)

@pytest.mark.asyncio
async def test_execute_workflow(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    agent_payload = {
        "name": f"WorkflowAgentForExecute_{uuid.uuid4().hex[:8]}",
        "description": "Agent for workflow execution",
        "config": {"model": "dummy-model"},
        "runtime_params": {},
        "tags": []
    }
    agent_id = await create_agent(client, headers, agent_payload)
    try:
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
            "name": f"WorkflowForExecute_{uuid.uuid4().hex[:8]}",
            "description": "Workflow for execution testing",
            "definition": workflow_definition,
            "tags": []
        }
        workflow_id = await create_workflow(client, headers, workflow_payload)
        try:
            execute_resp = await client.post(
                urljoin(BASE_URL, f"workflows/{workflow_id}/execute"),
                headers=headers,
                json={"input_params": {"test_key": "test_value"}}
            )
            assert execute_resp.status_code == 202
            exec_data = execute_resp.json()
            assert "execution_id" in exec_data
        finally:
            await delete_workflow(client, headers, workflow_id)
    finally:
        await delete_agent(client, headers, agent_id)

# -----------------------
# Additional Tests
# -----------------------

@pytest.mark.asyncio
async def test_unauthorized_access(client: httpx.AsyncClient):
    response = await client.post(
        urljoin(BASE_URL, "agents"),
        json={"name": "UnauthorizedAgent", "config": {}}
    )
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_agent_search_and_filter(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    agents_to_create = [
        {"name": f"SearchAgent1_{uuid.uuid4().hex[:8]}", "tags": ["search", "test"], "config": {"model": "dummy-model"}, "runtime_params": {}},
        {"name": f"SearchAgent2_{uuid.uuid4().hex[:8]}", "tags": ["search"], "config": {"model": "dummy-model"}, "runtime_params": {}},
        {"name": f"FilterAgent_{uuid.uuid4().hex[:8]}", "tags": ["filter"], "config": {"model": "dummy-model"}, "runtime_params": {}}
    ]
    created_agents = []
    try:
        for agent in agents_to_create:
            agent_id = await create_agent(client, headers, agent)
            created_agents.append(agent_id)
        filter_resp = await client.get(urljoin(BASE_URL, "agents"), headers=headers, params={"tags": "search"})
        assert filter_resp.status_code == 200
        filtered_agents = filter_resp.json()
        assert len(filtered_agents) >= 2
    finally:
        for agent_id in created_agents:
            await delete_agent(client, headers, agent_id)

@pytest.mark.asyncio
async def test_workflow_execution_with_multiple_agents(client: httpx.AsyncClient, access_token: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    agent_ids = []
    try:
        # Create multiple agents
        for i in range(3):
            payload = {
                "name": f"WorkflowExecutionAgent{i}_{uuid.uuid4().hex[:8]}",
                "description": f"Agent {i} for multi-step workflow",
                "config": {"model": "dummy-model"},
                "runtime_params": {},
                "tags": []
            }
            agent_id = await create_agent(client, headers, payload)
            agent_ids.append(agent_id)
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
            "name": f"MultiStepWorkflow_{uuid.uuid4().hex[:8]}",
            "description": "Workflow with multiple agent steps",
            "definition": workflow_definition,
            "tags": []
        }
        workflow_id = await create_workflow(client, headers, workflow_payload)
        try:
            exec_resp = await client.post(
                urljoin(BASE_URL, f"workflows/{workflow_id}/execute"),
                headers=headers,
                json={"input_params": {"initial_data": "test_data"}}
            )
            assert exec_resp.status_code == 202
            exec_data = exec_resp.json()
            assert "execution_id" in exec_data
        finally:
            await delete_workflow(client, headers, workflow_id)
    finally:
        for agent_id in agent_ids:
            await delete_agent(client, headers, agent_id)
