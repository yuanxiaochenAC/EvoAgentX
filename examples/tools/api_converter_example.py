import json
from evoagentx.tools.api_converter import (
    OpenAPIConverter,
    RapidAPIConverter,
    create_openapi_toolkit,
    create_rapidapi_toolkit
)

# Example 1: Create a toolkit using the OpenAPI specification
def example_openapi_converter():
    """Example of creating a toolkit using the OpenAPI specification"""
    
    # Example OpenAPI specification (simplified version)
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Weather API",
            "version": "1.0.0",
            "description": "A simple weather API"
        },
        "servers": [
            {
                "url": "https://api.weather.com/v1"
            }
        ],
        "paths": {
            "/weather": {
                "get": {
                    "operationId": "getCurrentWeather",
                    "summary": "Get current weather",
                    "description": "Get current weather for a specific location",
                    "parameters": [
                        {
                            "name": "city",
                            "in": "query",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "City name"
                        },
                        {
                            "name": "units",
                            "in": "query",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["metric", "imperial"]
                            },
                            "description": "Temperature units"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Weather data",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "temperature": {"type": "number"},
                                            "humidity": {"type": "number"},
                                            "description": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/forecast/{days}": {
                "get": {
                    "operationId": "getWeatherForecast",
                    "summary": "Get weather forecast",
                    "description": "Get weather forecast for specified number of days",
                    "parameters": [
                        {
                            "name": "days",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10
                            },
                            "description": "Number of forecast days"
                        },
                        {
                            "name": "city",
                            "in": "query",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "City name"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Forecast data"
                        }
                    }
                }
            }
        }
    }
    
    # Method 1: Using converter class
    print("=== Using OpenAPIConverter ===")
    converter = OpenAPIConverter(
        input_schema=openapi_spec,
        description="Weather service API",
        auth_config={
            "api_key": "your-api-key",
            "key_name": "X-API-Key"
        }
    )
    
    toolkit = converter.convert_to_toolkit()
    print(f"Service name: {toolkit.name}")
    print(f"Base URL: {toolkit.base_url}")
    print(f"Number of tools: {len(toolkit.tools)}")
    
    for tool in toolkit.tools:
        print(f"\nTool: {tool.name}")
        print(f"Description: {tool.description}")
        print(f"Input parameters: {list(tool.inputs.keys())}")
        print(f"Required parameters: {tool.required}")
    
    # Method 2: Using utility function
    print("\n=== Using utility function ===")
    toolkit2 = create_openapi_toolkit(
        schema_path_or_dict=openapi_spec,
        service_name="Weather Service",
        auth_config={"api_key": "your-api-key"}
    )
    print(f"Toolkit created by utility function: {toolkit2.name}")
    
    return toolkit


# Example 2: Create a toolkit using RapidAPI
def example_rapidapi_converter():
    """Example of creating a toolkit using RapidAPI"""
    
    # OpenAPI specification for RapidAPI is usually obtained from RapidAPI platform
    rapidapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Translation API",
            "version": "1.0.0"
        },
        "servers": [
            {
                "url": "https://microsoft-translator-text.p.rapidapi.com"
            }
        ],
        "paths": {
            "/translate": {
                "post": {
                    "operationId": "translateText",
                    "summary": "Translate text",
                    "description": "Translate text from one language to another",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "Text to translate"
                                        },
                                        "from": {
                                            "type": "string",
                                            "description": "Source language code"
                                        },
                                        "to": {
                                            "type": "string",
                                            "description": "Target language code"
                                        }
                                    },
                                    "required": ["text", "to"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Translation result"
                        }
                    }
                }
            }
        }
    }
    
    print("\n=== Using RapidAPIConverter ===")
    
    # Method 1: Using converter class
    converter = RapidAPIConverter(
        input_schema=rapidapi_spec,
        description="Microsoft Translator API",
        rapidapi_key="your-rapidapi-key",
        rapidapi_host="microsoft-translator-text.p.rapidapi.com"
    )
    
    toolkit = converter.convert_to_toolkit()
    print(f"RapidAPI service: {toolkit.name}")
    print(f"Common headers: {toolkit.common_headers}")
    
    for tool in toolkit.tools:
        print(f"\nTool: {tool.name}")
        print(f"Description: {tool.description}")
        print(f"Input parameters: {list(tool.inputs.keys())}")
    
    # Method 2: Using utility function
    toolkit2 = create_rapidapi_toolkit(
        schema_path_or_dict=rapidapi_spec,
        rapidapi_key="your-rapidapi-key",
        rapidapi_host="microsoft-translator-text.p.rapidapi.com",
        service_name="Translation Service"
    )
    
    return toolkit


# Example 3: Use the API toolkit in CustomizeAgent
def example_with_customize_agent():
    """Demonstrate how to use the API toolkit in CustomizeAgent"""
    
    # First create API toolkit
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {"title": "Calculator API", "version": "1.0.0"},
        "servers": [{"url": "https://api.calculator.com/v1"}],
        "paths": {
            "/calculate": {
                "post": {
                    "operationId": "calculate",
                    "summary": "Perform calculation",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "expression": {
                                            "type": "string",
                                            "description": "Mathematical expression to calculate"
                                        }
                                    },
                                    "required": ["expression"]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    # Create API toolkit
    api_toolkit = create_openapi_toolkit(
        schema_path_or_dict=openapi_spec,
        auth_config={"api_key": "calc-api-key"}
    )
    
    print("\n=== Using API tools in CustomizeAgent ===")
    print(f"API toolkit: {api_toolkit.name}")
    print(f"Available tools: {[tool.name for tool in api_toolkit.tools]}")
    
    # Note: Import CustomizeAgent when actually using
    # from evoagentx.agents.customize_agent import CustomizeAgent
    # 
    # agent = CustomizeAgent(
    #     name="Calculator Agent",
    #     description="An agent that can perform calculations using external API",
    #     prompt="Calculate the result of: {expression}",
    #     inputs=[
    #         {"name": "expression", "type": "string", "description": "Mathematical expression"}
    #     ],
    #     outputs=[
    #         {"name": "result", "type": "string", "description": "Calculation result"}
    #     ],
    #     tools=[api_toolkit]  # Use API toolkit
    # )
    
    return api_toolkit


# Example 4: Load API specification from file
def example_load_from_file():
    """Example of loading API specification from file"""
    
    # Create sample OpenAPI file
    sample_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Sample API",
            "version": "1.0.0"
        },
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "listUsers",
                    "summary": "List users",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {"type": "integer"},
                            "description": "Number of users to return"
                        }
                    ]
                }
            }
        }
    }
    
    # Save to file
    spec_file = "/tmp/sample_api_spec.json"
    with open(spec_file, 'w') as f:
        json.dump(sample_spec, f, indent=2)
    
    print(f"\n=== Load API specification from file ===")
    print(f"Spec file: {spec_file}")
    
    # Create toolkit from file
    toolkit = create_openapi_toolkit(
        schema_path_or_dict=spec_file,
        service_name="Sample Service"
    )
    
    print(f"Toolkit created from file: {toolkit.name}")
    print(f"Tools: {[tool.name for tool in toolkit.tools]}")
    
    return toolkit


if __name__ == "__main__":
    print("API Converter usage examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_openapi_converter()
        example_rapidapi_converter()
        example_with_customize_agent()
        example_load_from_file()
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
