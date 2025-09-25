import os
import json
from typing import Dict, Any
from evoagentx.tools.api_converter import create_openapi_toolkit, create_rapidapi_toolkit

# Example 1: OpenWeatherMap API integration
def create_weather_toolkit():
    """Create a weather API toolkit"""
    
    # Simplified OpenAPI specification for OpenWeatherMap API
    weather_api_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "OpenWeatherMap API",
            "version": "2.5",
            "description": "Weather data API"
        },
        "servers": [
            {"url": "https://api.openweathermap.org/data/2.5"}
        ],
        "paths": {
            "/weather": {
                "get": {
                    "operationId": "getCurrentWeather",
                    "summary": "Get current weather data",
                    "description": "Access current weather data for any location",
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "City name, state code and country code divided by comma"
                        },
                        {
                            "name": "appid",
                            "in": "query", 
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Your unique API key"
                        },
                        {
                            "name": "units",
                            "in": "query",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["standard", "metric", "imperial"]
                            },
                            "description": "Units of measurement"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "main": {
                                                "type": "object",
                                                "properties": {
                                                    "temp": {"type": "number"},
                                                    "humidity": {"type": "number"}
                                                }
                                            },
                                            "weather": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "main": {"type": "string"},
                                                        "description": {"type": "string"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/forecast": {
                "get": {
                    "operationId": "getWeatherForecast",
                    "summary": "Get 5 day weather forecast",
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "City name"
                        },
                        {
                            "name": "appid",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "API key"
                        }
                    ],
                    "responses": {
                        "200": {"description": "Forecast data"}
                    }
                }
            }
        }
    }
    
    # Get API key from environment variables
    api_key = os.getenv("OPENWEATHER_API_KEY", "your-api-key")
    
    # Create toolkit with API key as default parameter
    toolkit = create_openapi_toolkit(
        schema_path_or_dict=weather_api_spec,
        auth_config={"api_key": api_key}
    )
    
    # Add default API key parameter for each tool
    for tool in toolkit.tools:
        original_function = tool.function
        
        def create_wrapper(func, api_key):
            def wrapper(**kwargs):
                kwargs.setdefault("appid", api_key)
                return func(**kwargs)
            return wrapper
        
        tool.function = create_wrapper(original_function, api_key)
    
    return toolkit


# Example 2: RapidAPI weather service integration (full three features)
def create_rapidapi_weather_toolkit():
    """Create a weather API toolkit (using RapidAPI) - includes three features"""
    
    # Complete specification for Open Weather13 on RapidAPI
    weather_api_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Open Weather13 API",
            "version": "2.0",
            "description": "Complete weather data API via RapidAPI with 3 endpoints"
        },
        "servers": [
            {"url": "https://open-weather13.p.rapidapi.com"}
        ],
        "paths": {
            "/city": {
                "get": {
                    "operationId": "getCityWeather",
                    "summary": "Get weather by city name",
                    "description": "Get current weather data by city name",
                    "parameters": [
                        {
                            "name": "city",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "City name, e.g.: new york, beijing, tokyo"
                        },
                        {
                            "name": "lang",
                            "in": "query", 
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["AF", "AL", "AR", "AZ", "BG", "CA", "CZ", "DA", "DE", "EL", "EN", "EU", "FA", "FI", "FR", "GL", "HE", "HI", "HR", "HU", "ID", "IT", "JA", "KR", "LA", "LT", "MK", "NO", "NL", "PL", "PT", "PT_BR", "RO", "RU", "SE", "SK", "SL", "SP", "ES", "SR", "TH", "TR", "UK", "VI", "ZH_CN", "ZH_TW", "ZU"],
                                "default": "EN"
                            },
                            "description": "Language code, supports Simplified Chinese (ZH_CN), Traditional Chinese (ZH_TW), and many other languages"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "City weather data response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "coord": {
                                                "type": "object",
                                                "properties": {
                                                    "lon": {"type": "number"},
                                                    "lat": {"type": "number"}
                                                }
                                            },
                                            "weather": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "id": {"type": "integer"},
                                                        "main": {"type": "string"},
                                                        "description": {"type": "string"},
                                                        "icon": {"type": "string"}
                                                    }
                                                }
                                            },
                                            "main": {
                                                "type": "object",
                                                "properties": {
                                                    "temp": {"type": "number"},
                                                    "feels_like": {"type": "number"},
                                                    "temp_min": {"type": "number"},
                                                    "temp_max": {"type": "number"},
                                                    "pressure": {"type": "number"},
                                                    "humidity": {"type": "number"},
                                                    "sea_level": {"type": "number"},
                                                    "grnd_level": {"type": "number"}
                                                }
                                            },
                                            "visibility": {"type": "number"},
                                            "wind": {
                                                "type": "object",
                                                "properties": {
                                                    "speed": {"type": "number"},
                                                    "deg": {"type": "number"}
                                                }
                                            },
                                            "clouds": {
                                                "type": "object",
                                                "properties": {
                                                    "all": {"type": "number"}
                                                }
                                            },
                                            "name": {"type": "string"},
                                            "sys": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {"type": "integer"},
                                                    "id": {"type": "integer"},
                                                    "country": {"type": "string"},
                                                    "sunrise": {"type": "integer"},
                                                    "sunset": {"type": "integer"}
                                                }
                                            },
                                            "timezone": {"type": "integer"},
                                            "id": {"type": "integer"},
                                            "cod": {"type": "integer"},
                                            "dt": {"type": "integer"},
                                            "base": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/latlon": {
                "get": {
                    "operationId": "getWeatherByCoordinates",
                    "summary": "Get weather by coordinates",
                    "description": "Get current weather data using latitude and longitude",
                    "parameters": [
                        {
                            "name": "latitude",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Latitude, e.g.: 40.730610"
                        },
                        {
                            "name": "longitude",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Longitude, e.g.: -73.935242"
                        },
                        {
                            "name": "lang",
                            "in": "query", 
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["AF", "AL", "AR", "AZ", "BG", "CA", "CZ", "DA", "DE", "EL", "EN", "EU", "FA", "FI", "FR", "GL", "HE", "HI", "HR", "HU", "ID", "IT", "JA", "KR", "LA", "LT", "MK", "NO", "NL", "PL", "PT", "PT_BR", "RO", "RU", "SE", "SK", "SL", "SP", "ES", "SR", "TH", "TR", "UK", "VI", "ZH_CN", "ZH_TW", "ZU"],
                                "default": "EN"
                            },
                            "description": "Language code"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Coordinate weather data response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "coord": {
                                                "type": "object",
                                                "properties": {
                                                    "lon": {"type": "number"},
                                                    "lat": {"type": "number"}
                                                }
                                            },
                                            "weather": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "id": {"type": "integer"},
                                                        "main": {"type": "string"},
                                                        "description": {"type": "string"},
                                                        "icon": {"type": "string"}
                                                    }
                                                }
                                            },
                                            "main": {
                                                "type": "object",
                                                "properties": {
                                                    "temp": {"type": "number"},
                                                    "feels_like": {"type": "number"},
                                                    "temp_min": {"type": "number"},
                                                    "temp_max": {"type": "number"},
                                                    "pressure": {"type": "number"},
                                                    "humidity": {"type": "number"},
                                                    "sea_level": {"type": "number"},
                                                    "grnd_level": {"type": "number"}
                                                }
                                            },
                                            "visibility": {"type": "number"},
                                            "wind": {
                                                "type": "object",
                                                "properties": {
                                                    "speed": {"type": "number"},
                                                    "deg": {"type": "number"}
                                                }
                                            },
                                            "clouds": {
                                                "type": "object",
                                                "properties": {
                                                    "all": {"type": "number"}
                                                }
                                            },
                                            "name": {"type": "string"},
                                            "sys": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {"type": "integer"},
                                                    "id": {"type": "integer"},
                                                    "country": {"type": "string"},
                                                    "sunrise": {"type": "integer"},
                                                    "sunset": {"type": "integer"}
                                                }
                                            },
                                            "timezone": {"type": "integer"},
                                            "id": {"type": "integer"},
                                            "cod": {"type": "integer"},
                                            "dt": {"type": "integer"},
                                            "base": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/fivedaysforcast": {
                "get": {
                    "operationId": "getFiveDayForecast",
                    "summary": "Get 5-day weather forecast",
                    "description": "Get 5-day weather forecast using latitude and longitude",
                    "parameters": [
                        {
                            "name": "latitude",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Latitude, e.g.: 40.730610"
                        },
                        {
                            "name": "longitude",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Longitude, e.g.: -73.935242"
                        },
                        {
                            "name": "lang",
                            "in": "query", 
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["AF", "AL", "AR", "AZ", "BG", "CA", "CZ", "DA", "DE", "EL", "EN", "EU", "FA", "FI", "FR", "GL", "HE", "HI", "HR", "HU", "ID", "IT", "JA", "KR", "LA", "LT", "MK", "NO", "NL", "PL", "PT", "PT_BR", "RO", "RU", "SE", "SK", "SL", "SP", "ES", "SR", "TH", "TR", "UK", "VI", "ZH_CN", "ZH_TW", "ZU"],
                                "default": "EN"
                            },
                            "description": "Language code"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "5-day forecast data response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "cod": {"type": "string"},
                                            "message": {"type": "number"},
                                            "cnt": {"type": "integer"},
                                            "list": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "dt": {"type": "integer"},
                                                        "main": {
                                                            "type": "object",
                                                            "properties": {
                                                                "temp": {"type": "number"},
                                                                "feels_like": {"type": "number"},
                                                                "temp_min": {"type": "number"},
                                                                "temp_max": {"type": "number"},
                                                                "pressure": {"type": "number"},
                                                                "sea_level": {"type": "number"},
                                                                "grnd_level": {"type": "number"},
                                                                "humidity": {"type": "number"},
                                                                "temp_kf": {"type": "number"}
                                                            }
                                                        },
                                                        "weather": {
                                                            "type": "array",
                                                            "items": {
                                                                "type": "object",
                                                                "properties": {
                                                                    "id": {"type": "integer"},
                                                                    "main": {"type": "string"},
                                                                    "description": {"type": "string"},
                                                                    "icon": {"type": "string"}
                                                                }
                                                            }
                                                        },
                                                        "clouds": {
                                                            "type": "object",
                                                            "properties": {
                                                                "all": {"type": "number"}
                                                            }
                                                        },
                                                        "wind": {
                                                            "type": "object",
                                                            "properties": {
                                                                "speed": {"type": "number"},
                                                                "deg": {"type": "number"},
                                                                "gust": {"type": "number"}
                                                            }
                                                        },
                                                        "visibility": {"type": "number"},
                                                        "pop": {"type": "number"},
                                                        "rain": {
                                                            "type": "object",
                                                            "properties": {
                                                                "3h": {"type": "number"}
                                                            }
                                                        },
                                                        "sys": {
                                                            "type": "object",
                                                            "properties": {
                                                                "pod": {"type": "string"}
                                                            }
                                                        },
                                                        "dt_txt": {"type": "string"}
                                                    }
                                                }
                                            },
                                            "city": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "integer"},
                                                    "name": {"type": "string"},
                                                    "coord": {
                                                        "type": "object",
                                                        "properties": {
                                                            "lat": {"type": "number"},
                                                            "lon": {"type": "number"}
                                                        }
                                                    },
                                                    "country": {"type": "string"},
                                                    "population": {"type": "integer"},
                                                    "timezone": {"type": "integer"},
                                                    "sunrise": {"type": "integer"},
                                                    "sunset": {"type": "integer"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    # Get RapidAPI credentials from environment variables
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    rapidapi_host = "open-weather13.p.rapidapi.com"
    
    toolkit = create_rapidapi_toolkit(
        schema_path_or_dict=weather_api_spec,
        rapidapi_key=rapidapi_key,
        rapidapi_host=rapidapi_host,
        service_name="Open Weather13"
    )
    
    return toolkit


# Example 3: Create multi-service intelligent agent
def create_multi_service_agent():
    """Create an intelligent agent that uses multiple API services"""
    
    try:
        from evoagentx.agents.customize_agent import CustomizeAgent
        from evoagentx.models.model_configs import LLMConfig
        
        # Create various API toolkits
        weather_toolkit = create_weather_toolkit()
        rapidapi_weather_toolkit = create_rapidapi_weather_toolkit()
        
        # Create intelligent agent, integrating multiple API services
        agent = CustomizeAgent(
            name="Multi-Service Assistant",
            description="An intelligent assistant that can provide weather information from multiple sources",
            prompt="""
You are a helpful assistant with access to multiple weather APIs.

User request: {user_request}

Please help the user by:
1. Understanding what they need
2. Using the appropriate API tools
3. Providing a clear and helpful response

Available tools:
- OpenWeatherMap tools: getCurrentWeather, getWeatherForecast
- RapidAPI Weather tools: getCityWeather
            """,
            inputs=[
                {
                    "name": "user_request",
                    "type": "string", 
                    "description": "The user's request for weather services"
                }
            ],
            outputs=[
                {
                    "name": "response",
                    "type": "string",
                    "description": "The assistant's response with requested information"
                }
            ],
            tools=[weather_toolkit, rapidapi_weather_toolkit],  # Use multiple toolkits
            max_tool_calls=3,
            # llm_config=LLMConfig(model="gpt-3.5-turbo")  # Need to configure LLM
        )
        
        return agent
        
    except ImportError as e:
        print(f"Failed to import CustomizeAgent: {e}")
        print("Please ensure the EvoAgentX framework is correctly installed")
        return None


# Example 4: Advanced configuration for toolkits
def create_advanced_api_toolkit():
    """Create an API toolkit with advanced configuration"""
    
    # API specification with complex authentication
    advanced_api_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Advanced API",
            "version": "1.0.0"
        },
        "servers": [{"url": "https://api.advanced-service.com/v1"}],
        "paths": {
            "/data": {
                "get": {
                    "operationId": "getData",
                    "summary": "Get data",
                    "parameters": [
                        {
                            "name": "filter",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Data filter"
                        }
                    ],
                    "responses": {"200": {"description": "Data response"}}
                }
            }
        }
    }
    
    # Create a toolkit with custom configuration
    toolkit = create_openapi_toolkit(
        schema_path_or_dict=advanced_api_spec,
        auth_config={
            "api_key": "your-api-key",
            "bearer_token": "your-bearer-token"  # Support multiple authentication methods
        }
    )
    
    # Add custom headers
    toolkit.common_headers.update({
        "User-Agent": "EvoAgentX-Client/1.0",
        "Accept": "application/json",
        "Cache-Control": "no-cache"
    })
    
    # Add custom behavior for tools
    for tool in toolkit.tools:
        original_function = tool.function
        
        def create_enhanced_wrapper(func):
            def enhanced_wrapper(**kwargs):
                # Pre-request processing
                print(f"Calling API tool: {tool.name}")
                print(f"Parameters: {kwargs}")
                
                try:
                    result = func(**kwargs)
                    # Post-response processing
                    print(f"API call succeeded, return type: {type(result)}")
                    return result
                except Exception as e:
                    print(f"API call failed: {e}")
                    # You can add retry logic or error handling here
                    raise
                    
            return enhanced_wrapper
        
        tool.function = create_enhanced_wrapper(original_function)
    
    return toolkit


# Weather API feature demo function
def demo_weather_functions():
    """Demonstrate detailed usage of the three weather API features"""
    print("=== Demonstration of three weather API features ===\n")
    
    try:
        # Create toolkit
        toolkit = create_rapidapi_weather_toolkit()
        print(f"Toolkit created successfully with {len(toolkit.tools)} tools:")
        for tool in toolkit.tools:
            print(f"  - {tool.name}: {tool.description}")
        print()
        
        # Get tool
        city_tool = None
        coord_tool = None
        forecast_tool = None
        
        for tool in toolkit.tools:
            if "getCityWeather" in tool.name:
                city_tool = tool
            elif "getWeatherByCoordinates" in tool.name:
                coord_tool = tool
            elif "getFiveDayForecast" in tool.name:
                forecast_tool = tool
        
        # Feature 1: Query weather by city name
        print("[Feature 1] Query weather by city name")
        print("-" * 40)
        if city_tool:
            cities = ["new york", "beijing", "tokyo"]
            for city in cities:
                try:
                    print(f"Querying weather for {city}...")
                    result = city_tool.function(city=city, lang="EN")
                    
                    name = result.get('name', 'Unknown')
                    country = result.get('sys', {}).get('country', 'Unknown')
                    temp = result.get('main', {}).get('temp', 'N/A')
                    humidity = result.get('main', {}).get('humidity', 'N/A')
                    weather = result.get('weather', [{}])[0].get('description', 'N/A')
                    
                    print(f"  ✓ {name}, {country}")
                    print(f"    Temperature: {temp}°F, Humidity: {humidity}%")
                    print(f"    Weather: {weather}")
                    print()
                except Exception as e:
                    print(f"  ✗ Failed to query {city}: {e}")
                    print()
        
        # Feature 2: Query weather by coordinates
        print("[Feature 2] Query weather by coordinates")
        print("-" * 40)
        if coord_tool:
            coordinates = [
                ("40.730610", "-73.935242", "New York area"),
                ("39.904200", "116.407396", "Beijing area"),
                ("35.689487", "139.691706", "Tokyo area")
            ]
            
            for lat, lon, desc in coordinates:
                try:
                    print(f"Querying weather for {desc} ({lat}, {lon})...")
                    result = coord_tool.function(latitude=lat, longitude=lon, lang="EN")
                    
                    name = result.get('name', 'Unknown')
                    country = result.get('sys', {}).get('country', 'Unknown')
                    temp = result.get('main', {}).get('temp', 'N/A')
                    humidity = result.get('main', {}).get('humidity', 'N/A')
                    weather = result.get('weather', [{}])[0].get('description', 'N/A')
                    
                    print(f"  ✓ {name}, {country}")
                    print(f"    Temperature: {temp}K, Humidity: {humidity}%")
                    print(f"    Weather: {weather}")
                    print()
                except Exception as e:
                    print(f"  ✗ Failed to query {desc}: {e}")
                    print()
        
        # Feature 3: 5-day weather forecast
        print("[Feature 3] 5-day weather forecast")
        print("-" * 40)
        if forecast_tool:
            try:
                print("Getting 5-day weather forecast for New York area...")
                result = forecast_tool.function(latitude="40.730610", longitude="-73.935242", lang="EN")
                
                city_info = result.get('city', {})
                city_name = city_info.get('name', 'Unknown')
                country = city_info.get('country', 'Unknown')
                cnt = result.get('cnt', 0)
                
                print(f"  ✓ {city_name}, {country} - Total {cnt} forecast entries")
                
                forecasts = result.get('list', [])
                if forecasts:
                    print("  Weather forecast for the next few days:")
                    for i, forecast in enumerate(forecasts):  # Show the first 10 entries
                        dt_txt = forecast.get('dt_txt', 'Unknown')
                        temp = forecast.get('main', {}).get('temp', 'N/A')
                        weather_desc = forecast.get('weather', [{}])[0].get('description', 'N/A')
                        humidity = forecast.get('main', {}).get('humidity', 'N/A')
                        
                        print(f"    {dt_txt}: {temp}K, {weather_desc}, Humidity {humidity}%")
                        
                        if 'rain' in forecast:
                            rain = forecast['rain'].get('3h', 0)
                            if rain > 0:
                                print(f"      Rainfall: {rain}mm")
                print()
            except Exception as e:
                print(f"  ✗ Failed to get 5-day forecast: {e}")
                print()
        
    except Exception as e:
        print(f"Error occurred during demonstration: {e}")
    
    print("=== Demonstration completed ===")


# Usage example
def main():
    """Main function, demonstrate various API integrations"""
    
    print("=== API Converter Real World Examples ===\n")
    
    # # 1. Create weather toolkit
    # print("1. Creating weather API toolkit...")
    # try:
    #     weather_toolkit = create_weather_toolkit()
    #     print(f"   Weather toolkit created successfully: {weather_toolkit.name}")
    #     print(f"   Contains tools: {[tool.name for tool in weather_toolkit.tools]}")
        
    #     # Demonstrate weather tool usage (requires valid API key)
    #     if os.getenv("OPENWEATHER_API_KEY"):
    #         weather_tool = weather_toolkit.get_tool("getCurrentWeather")
    #         # result = weather_tool(q="Beijing,CN", units="metric")
    #         # print(f"   Beijing weather: {result}")
    #     else:
    #         print("   Tip: Set OPENWEATHER_API_KEY environment variable to test actual API calls")
            
    # except Exception as e:
    #     print(f"   Error creating weather toolkit: {e}")
    
    # print()
    
    # 2. Create RapidAPI weather toolkit (includes three features)
    print("2. Creating RapidAPI weather API toolkit (includes three features)...")
    try:
        rapidapi_weather_toolkit = create_rapidapi_weather_toolkit()
        print(f"   RapidAPI weather toolkit created successfully: {rapidapi_weather_toolkit.name}")
        print(f"   Contains tools: {[tool.name for tool in rapidapi_weather_toolkit.tools]}")
        print(f"   RapidAPI header configuration: {rapidapi_weather_toolkit.common_headers}")
        
        # Get all three tools
        city_tool = None
        latlon_tool = None
        forecast_tool = None
        
        for tool in rapidapi_weather_toolkit.tools:
            if tool.name == "getCityWeather":
                city_tool = tool
            elif tool.name == "getWeatherByCoordinates":
                latlon_tool = tool
            elif tool.name == "getFiveDayForecast":
                forecast_tool = tool
        
        print("\n   === Feature 1: Query weather by city name ===")
        if city_tool:
            try:
                print("   Querying New York weather...")
                result = city_tool.function(city="new york", lang="EN")
                print("   ✓ New York weather query successful!")
                print(f"   City: {result.get('name', 'N/A')}")
                if 'main' in result:
                    temp_f = result['main'].get('temp', 'N/A')
                    humidity = result['main'].get('humidity', 'N/A')
                    print(f"   Temperature: {temp_f}°F")
                    print(f"   Humidity: {humidity}%")
                if 'weather' in result and result['weather']:
                    weather_desc = result['weather'][0].get('description', 'N/A')
                    print(f"   Weather condition: {weather_desc}")
                if 'coord' in result:
                    lon = result['coord'].get('lon', 'N/A')
                    lat = result['coord'].get('lat', 'N/A')
                    print(f"   Coordinates: {lat}, {lon}")
            except Exception as api_error:
                print(f"   ✗ City weather query failed: {api_error}")
        
        print("\n   === Feature 2: Query weather by coordinates ===")
        if latlon_tool:
            try:
                print("   Querying weather for coordinates (40.730610, -73.935242)...")
                result = latlon_tool.function(latitude="40.730610", longitude="-73.935242", lang="EN")
                print("   ✓ Coordinate weather query successful!")
                print(f"   Location: {result.get('name', 'N/A')}")
                if 'main' in result:
                    temp = result['main'].get('temp', 'N/A')
                    humidity = result['main'].get('humidity', 'N/A')
                    print(f"   Temperature: {temp}K")
                    print(f"   Humidity: {humidity}%")
                if 'weather' in result and result['weather']:
                    weather_desc = result['weather'][0].get('description', 'N/A')
                    print(f"   Weather condition: {weather_desc}")
            except Exception as api_error:
                print(f"   ✗ Coordinate weather query failed: {api_error}")
        
        print("\n   === Feature 3: 5-day weather forecast ===")
        if forecast_tool:
            try:
                print("   Getting 5-day weather forecast...")
                result = forecast_tool.function(latitude="40.730610", longitude="-73.935242", lang="EN")
                print("   ✓ 5-day weather forecast retrieved successfully!")
                print(f"   Forecast count: {result.get('cnt', 'N/A')}")
                if 'city' in result:
                    city_name = result['city'].get('name', 'N/A')
                    print(f"   City: {city_name}")
                if 'list' in result and result['list']:
                    print("   First 3 days forecast:")
                    for i, forecast in enumerate(result['list'][:3]):
                        dt_txt = forecast.get('dt_txt', 'N/A')
                        if 'main' in forecast:
                            temp = forecast['main'].get('temp', 'N/A')
                            print(f"     {dt_txt}: {temp}K")
                        if 'weather' in forecast and forecast['weather']:
                            desc = forecast['weather'][0].get('description', 'N/A')
                            print(f"       Weather: {desc}")
            except Exception as api_error:
                print(f"   ✗ Failed to get 5-day weather forecast: {api_error}")
            
    except Exception as e:
        print(f"   Error creating RapidAPI weather toolkit: {e}")
    
    print()
    
    # 4. Create multi-service agent
    # print("3. Creating multi-service intelligent agent...")
    # try:
    #     agent = create_multi_service_agent()
    #     if agent:
    #         print(f"   Multi-service agent created successfully: {agent.name}")
    #         print(f"   Agent description: {agent.description}")
            
    #         # Demonstrate agent usage
    #         # result = agent(inputs={"user_request": "What's the weather like in Tokyo?"})
    #         # print(f"   Agent response: {result}")
    #     else:
    #         print("   Multi-service agent creation failed (possibly missing dependencies)")
            
    # except Exception as e:
    #     print(f"   Error creating multi-service agent: {e}")
    
    # print()
    
    # # 4. Create advanced configuration toolkit
    # print("4. Creating advanced configuration API toolkit...")
    # try:
    #     advanced_toolkit = create_advanced_api_toolkit()
    #     print(f"   Advanced toolkit created successfully: {advanced_toolkit.name}")
    #     print(f"   Custom request headers: {advanced_toolkit.common_headers}")
        
    # except Exception as e:
    #     print(f"   Error creating advanced toolkit: {e}")
    
    print("\n=== Basic examples completed ===")
    print("\nNow demonstrating complete weather API three major features...")
    print("=" * 60)
    
    # Demonstrate three weather API features
    demo_weather_functions()


if __name__ == "__main__":
    main()
