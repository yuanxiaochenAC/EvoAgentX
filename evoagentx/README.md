# Agent API

This project provides a FastAPI-based server for storing agents and workflows, with token-based access management.

## Features

- Store and manage agent configurations, states, and metadata
- Define and execute workflows that orchestrate agents
- Secure API endpoints with JWT token-based authentication
- Comprehensive API documentation with Swagger UI and ReDoc
- Docker-based deployment for easy setup

## Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- Poetry (for local development)

### Using Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/agent-api.git
   cd agent-api
   ```

2. Create a `.env` file with your configuration:
   ```bash
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_secure_password
   POSTGRES_DB=agent_api
   SECRET_KEY=your_secret_key_for_jwt
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. The API will be available at `http://localhost:8000`

### Local Development

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Create a `.env` file with your configuration.

3. Run the application:
   ```bash
   poetry run uvicorn app.main:app --reload
   ```

## Usage

### Authentication

1. Create a user:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/users/" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "secure_password"}'
   ```

2. Obtain an access token:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=user@example.com&password=secure_password"
   ```

3. Use the access token for authenticated requests:
   ```bash
   curl -X GET "http://localhost:8000/api/v1/agents/" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
   ```

### API Documentation

- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc`

## Testing

Run the tests using pytest:

```bash
poetry run pytest
```

## Project Structure

- `app/api/`: API endpoints
- `app/core/`: Core settings and security
- `app/db/`: Database models and session management
- `app/schemas/`: Pydantic models for request/response validation
- `app/services/`: Business logic
- `tests/`: Unit and integration tests

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
