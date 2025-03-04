import pytest
import httpx
import requests
from fastapi.testclient import TestClient
from evoagentx.app.main import app
from evoagentx.app.db import Database
from evoagentx.app.security import create_user, get_password_hash
from evoagentx.app.config import settings
from datetime import datetime

# Test user credentials
TEST_USER = {
    "username": "testuser@example.com",
    "password": "testpassword"
}

@pytest.fixture(autouse=True, scope="session")
async def setup_test_db():
    """Initialize test database"""
    # Connect to test database
    await Database.connect(settings.MONGODB_URL, f"{settings.DATABASE_NAME}_test")
    
    # Create test user
    test_user = {
        "email": TEST_USER["username"],
        "hashed_password": get_password_hash(TEST_USER["password"]),
        "is_active": True,
        "is_admin": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    # Clear existing test data
    await Database.db.users.delete_many({})
    await Database.db.agents.delete_many({})
    await Database.db.workflows.delete_many({})
    
    # Insert test user
    await Database.db.users.insert_one(test_user)
    
    yield
    
    # Cleanup after all tests
    await Database.db.users.delete_many({})
    await Database.db.agents.delete_many({})
    await Database.db.workflows.delete_many({})
    await Database.disconnect()

@pytest.fixture
def test_client():
    """Create a test client with the app"""
    return TestClient(app)

# Helper function to get auth token
async def get_auth_token(client):
    """Get authentication token for test user"""
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        }
    )
    assert response.status_code == 200
    data = response.json()
    return data["access_token"]

@pytest.mark.asyncio
async def test_httpx_async_request():
    """Async HTTP request using httpx"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        # First get auth token
        auth_response = await client.post(
            '/api/v1/auth/login',
            data=TEST_USER
        )
        assert auth_response.status_code == 200
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # GET request with auth
        response = await client.get('/api/v1/agents', headers=headers)
        assert response.status_code == 200

def test_requests_sync_request(test_client):
    """Synchronous HTTP request using test client"""
    # First get auth token
    auth_response = test_client.post(
        '/api/v1/auth/login',
        data=TEST_USER
    )
    assert auth_response.status_code == 200
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # GET request with auth
    response = test_client.get('/api/v1/agents', headers=headers)
    assert response.status_code == 200

@pytest.fixture
async def authenticated_client():
    """Fixture to create a reusable authenticated HTTP client"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        token = await get_auth_token(client)
        client.headers = {"Authorization": f"Bearer {token}"}
        yield client

@pytest.mark.asyncio
async def test_with_fixture(authenticated_client):
    """Using a fixture for HTTP requests"""
    response = await authenticated_client.get('/api/v1/agents')
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_authenticated_request():
    """Test an authenticated API endpoint"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        # Login to get token
        auth_response = await client.post(
            '/api/v1/auth/login',
            data=TEST_USER
        )
        assert auth_response.status_code == 200
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Make authenticated request
        response = await client.get('/api/v1/agents', headers=headers)
        assert response.status_code == 200

def test_fastapi_test_client(test_client):
    """Testing using FastAPI's TestClient"""
    # Get auth token
    auth_response = test_client.post('/api/v1/auth/login', data=TEST_USER)
    assert auth_response.status_code == 200
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # GET request with auth
    response = test_client.get('/api/v1/agents', headers=headers)
    assert response.status_code == 200
