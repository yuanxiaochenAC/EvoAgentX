# API Converter User Guide

The API Converter is a powerful tool in the EvoAgentX framework that automatically converts various API specifications (such as OpenAPI/Swagger and RapidAPI) into toolkits that can be used by intelligent agents.

## Core Components

### 1. APITool
A tool wrapper for a single API endpoint, inheriting from the Tool class.

Key features:
- Automatically handles API requests and responses
- Supports various HTTP methods (GET, POST, PUT, DELETE, PATCH)
- Flexible parameter handling (path params, query params, request body)
- Built-in error handling and result processing

### 2. APIToolkit
A collection of tools for an API service, inheriting from the Toolkit class.

Key features:
- Manage multiple related API tools
- Unified authentication configuration
- Common request header management
- Service-level configuration

### 3. BaseAPIConverter
Abstract base class for API converters that defines the conversion interface.

Core methods:
- convert_to_toolkit(): Convert an API specification to an APIToolkit
- _create_api_function(): Create an execution function for a single endpoint
- _extract_parameters(): Extract parameter information

### 4. OpenAPIConverter
OpenAPI (Swagger) specification converter that inherits from BaseAPIConverter.

Supported features:
- OpenAPI 3.0+ specifications
- Automatic parameter extraction
- Path parameter replacement
- Request body handling
- Response formatting

### 5. RapidAPIConverter
RapidAPI-specific converter that inherits from OpenAPIConverter.

RapidAPI features:
- Automatically add RapidAPI authentication headers
- Support RapidAPI host configuration
- Specialized error handling

## Usage

### Basic Usage

#### 1. Create a toolkit from an OpenAPI spec

```python
from evoagentx.tools.api_converter import create_openapi_toolkit

# Create from a dictionary
openapi_spec = {
    "openapi": "3.0.0",
    "info": {"title": "Weather API", "version": "1.0.0"},
    "servers": [{"url": "https://api.weather.com/v1"}],
    "paths": {
        "/weather": {
            "get": {
                "operationId": "getCurrentWeather",
                "summary": "Get current weather",
                "parameters": [
                    {
                        "name": "city",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "City name"
                    }
                ]
            }
        }
    }
}

toolkit = create_openapi_toolkit(
    schema_path_or_dict=openapi_spec,
    auth_config={"api_key": "your-api-key"}
)

# Create from a file
toolkit = create_openapi_toolkit(
    schema_path_or_dict="path/to/openapi.json",
    auth_config={"api_key": "your-api-key"}
)
```

#### 2. Create a RapidAPI toolkit

```python
from evoagentx.tools.api_converter import create_rapidapi_toolkit

toolkit = create_rapidapi_toolkit(
    schema_path_or_dict="path/to/rapidapi_spec.json",
    rapidapi_key="your-rapidapi-key",
    rapidapi_host="api.rapidapi.host.com"
)
```

#### 3. Use in CustomizeAgent

```python
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.tools.api_converter import create_openapi_toolkit

# Create the API toolkit
api_toolkit = create_openapi_toolkit(
    schema_path_or_dict=openapi_spec,
    auth_config={"api_key": "your-api-key"}
)

# Create an agent that uses the API toolkit
agent = CustomizeAgent(
    name="Weather Agent",
    description="An agent that provides weather information",
    prompt="Get weather information for {city}",
    inputs=[
        {"name": "city", "type": "string", "description": "City name"}
    ],
    outputs=[
        {"name": "weather_info", "type": "string", "description": "Weather information"}
    ],
    tools=[api_toolkit]
)

# Use the agent
result = agent(inputs={"city": "Beijing"})
```

### Advanced Usage

#### 1. Custom converter

```python
from evoagentx.tools.api_converter import BaseAPIConverter

class CustomAPIConverter(BaseAPIConverter):
    def convert_to_toolkit(self):
        # Implement custom conversion logic
        pass

    def _create_api_function(self, endpoint_config):
        # Implement custom API function creation
        pass
```

#### 2. Complex authentication configuration

```python
# API key auth
auth_config = {
    "api_key": "your-api-key",
    "key_name": "X-API-Key"  # Custom header name
}

# Bearer Token auth
auth_config = {
    "bearer_token": "your-bearer-token"
}

# Combined auth
auth_config = {
    "api_key": "your-api-key",
    "bearer_token": "your-bearer-token"
}
```

#### 3. Custom request headers

```python
from evoagentx.tools.api_converter import OpenAPIConverter

converter = OpenAPIConverter(
    input_schema=openapi_spec,
    auth_config=auth_config
)

toolkit = converter.convert_to_toolkit()

# Add custom request headers
toolkit.common_headers.update({
    "User-Agent": "MyApp/1.0",
    "Accept": "application/json"
})
```

## Supported API Specification Formats

### OpenAPI/Swagger
- Versions: OpenAPI 3.0+, Swagger 2.0
- Formats: JSON, YAML
- Features: Complete parameter extraction, request body handling, response formatting

### RapidAPI
- Based on: OpenAPI specification
- Enhancements: RapidAPI-specific authentication and configuration
- Features: Automatic RapidAPI header handling

## Parameter Type Mapping

| OpenAPI Type | Tool Input Type | Description |
|-------------|------------------|-------------|
| string      | string           | String      |
| integer     | integer          | Integer     |
| number      | number           | Number      |
| boolean     | boolean          | Boolean     |
| array       | array            | Array       |
| object      | object           | Object      |

## Error Handling

The API Converter provides layered error handling:

1. Conversion errors: Spec parsing failures, invalid formats
2. Runtime errors: API request failures, network issues
3. Response errors: API returns error status codes

```python
try:
    toolkit = create_openapi_toolkit(openapi_spec)
    result = toolkit.get_tool("weather_api")(city="Beijing")
except Exception as e:
    print(f"Error: {e}")
```

## Best Practices

### 1. API Key Management
```python
import os

# Store sensitive information in environment variables
api_key = os.getenv("WEATHER_API_KEY")
auth_config = {"api_key": api_key}
```

### 2. Retry on Errors
```python
import time
from functools import wraps

def retry_api_call(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
```

### 3. Response Caching
```python
from functools import lru_cache

# Cache results for stable API endpoints
@lru_cache(maxsize=100)
def cached_api_call(endpoint, params):
    # API call logic
    pass
```

## Troubleshooting

### Common Issues

1. Conversion failed
   - Check the OpenAPI spec format
   - Verify that required fields are present
   - Confirm server URL configuration

2. API call failed
   - Verify API key and authentication configuration
   - Check network connectivity
   - Confirm the API endpoint is accessible

3. Parameter errors
   - Check parameter names and types
   - Verify required parameters are provided
   - Confirm parameter value formats are correct

### Debugging Tips

```python
import logging

# Enable verbose logging
logging.basicConfig(level=logging.DEBUG)

# Inspect generated tools
for tool in toolkit.tools:
    print(f"Tool: {tool.name}")
    print(f"Inputs: {tool.inputs}")
    print(f"Required: {tool.required}")
```

## Extensibility

### Add support for a new API specification

1. Inherit from BaseAPIConverter
2. Implement the convert_to_toolkit() method
3. Implement the _create_api_function() method
4. Add spec-specific parameter extraction logic

### Customize tool behavior

1. Inherit from the APITool class
2. Override the __call__ method
3. Add custom result processing logic

## Example Project

See examples/tools/api_converter_example.py for a complete example, including:
- Basic OpenAPI conversion
- RapidAPI integration
- CustomizeAgent integration
- File loading example

---

With the API Converter, you can easily integrate any standards-compliant API into EvoAgentX agents, greatly expanding their capabilities and use cases.
