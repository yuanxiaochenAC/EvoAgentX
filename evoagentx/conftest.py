import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database, drop_database

from evoagentx.app.db.base import Base
from evoagentx.app.db.session import get_db
from evoagentx.main import app
from evoagentx.core.config import settings
from fastapi.testclient import TestClient
from evoagentx.core.security import get_password_hash
from evoagentx.db.models.user import User


# Test database URL - use an in-memory SQLite database for tests
TEST_SQLALCHEMY_DATABASE_URI = "sqlite:///./test.db"

# Create a test engine
engine = create_engine(TEST_SQLALCHEMY_DATABASE_URI)


@pytest.fixture(scope="session")
def db_engine():
    """
    Create a test database engine.
    """
    if database_exists(TEST_SQLALCHEMY_DATABASE_URI):
        drop_database(TEST_SQLALCHEMY_DATABASE_URI)
    
    create_database(TEST_SQLALCHEMY_DATABASE_URI)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Drop the test database after tests
    drop_database(TEST_SQLALCHEMY_DATABASE_URI)


@pytest.fixture(scope="function")
def db_session(db_engine):
    """
    Create a new database session for a test.
    """
    connection = db_engine.connect()
    # Start a transaction
    transaction = connection.begin()
    
    # Create a new session
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = TestSessionLocal()
    
    # Create a test user
    hashed_password = get_password_hash("testpassword")
    test_user = User(
        email="test@example.com",
        hashed_password=hashed_password,
        is_active=True,
    )
    session.add(test_user)
    session.commit()
    
    yield session
    
    # Close session and transaction
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db_session):
    """
    Create a test client with the test database.
    """
    # Override the get_db dependency to use the test session
    def _get_test_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = _get_test_db
    
    with TestClient(app) as c:
        yield c
    
    # Remove the override
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_user(db_session):
    """
    Return the test user from the database.
    """
    return db_session.query(User).filter(User.email == "test@example.com").first()


@pytest.fixture(scope="function")
def token_headers(client, test_user):
    """
    Get token headers for the test user.
    """
    login_data = {
        "username": test_user.email,
        "password": "testpassword",
    }
    response = client.post(f"{settings.API_V1_STR}/auth/login", data=login_data)
    tokens = response.json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}
