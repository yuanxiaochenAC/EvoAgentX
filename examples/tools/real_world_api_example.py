import os
import json
from typing import Dict, Any
from evoagentx.tools.api_converter import create_openapi_toolkit, create_rapidapi_toolkit

# 示例1：OpenWeatherMap API集成
def create_weather_toolkit():
    """创建天气API工具集"""
    
    # OpenWeatherMap API的简化OpenAPI规范
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
    
    # 从环境变量获取API密钥
    api_key = os.getenv("OPENWEATHER_API_KEY", "your-api-key")
    
    # 创建工具集，将API密钥作为默认参数
    toolkit = create_openapi_toolkit(
        schema_path_or_dict=weather_api_spec,
        auth_config={"api_key": api_key}
    )
    
    # 为每个工具添加默认的API密钥参数
    for tool in toolkit.tools:
        original_function = tool.function
        
        def create_wrapper(func, api_key):
            def wrapper(**kwargs):
                kwargs.setdefault("appid", api_key)
                return func(**kwargs)
            return wrapper
        
        tool.function = create_wrapper(original_function, api_key)
    
    return toolkit


# 示例2：RapidAPI天气服务集成（完整的三个功能）
def create_rapidapi_weather_toolkit():
    """创建天气API工具集（使用RapidAPI）- 包含三个功能"""
    
    # Open Weather13 on RapidAPI的完整规范
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
                    "summary": "根据城市名称获取天气",
                    "description": "通过城市名称获取当前天气数据",
                    "parameters": [
                        {
                            "name": "city",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "城市名称，例如：new york, beijing, tokyo"
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
                            "description": "语言代码，支持中文简体(ZH_CN)、中文繁体(ZH_TW)等多种语言"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "城市天气数据响应",
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
                    "summary": "根据经纬度获取天气",
                    "description": "通过经纬度坐标获取当前天气数据",
                    "parameters": [
                        {
                            "name": "latitude",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "纬度，例如：40.730610"
                        },
                        {
                            "name": "longitude",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "经度，例如：-73.935242"
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
                            "description": "语言代码"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "经纬度天气数据响应",
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
                    "summary": "获取5天天气预报",
                    "description": "通过经纬度坐标获取5天天气预报数据",
                    "parameters": [
                        {
                            "name": "latitude",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "纬度，例如：40.730610"
                        },
                        {
                            "name": "longitude",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "经度，例如：-73.935242"
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
                            "description": "语言代码"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "5天天气预报数据响应",
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
    
    # 从环境变量获取RapidAPI凭据
    rapidapi_key = os.getenv("RAPIDAPI_KEY", "5bad9b417amsh4fb61074f2d054bp114882jsn475094fc19bb")
    rapidapi_host = "open-weather13.p.rapidapi.com"
    
    toolkit = create_rapidapi_toolkit(
        schema_path_or_dict=weather_api_spec,
        rapidapi_key=rapidapi_key,
        rapidapi_host=rapidapi_host,
        service_name="Open Weather13"
    )
    
    return toolkit


# 示例3：创建多服务智能代理
def create_multi_service_agent():
    """创建使用多个API服务的智能代理"""
    
    try:
        from evoagentx.agents.customize_agent import CustomizeAgent
        from evoagentx.models.model_configs import LLMConfig
        
        # 创建各种API工具集
        weather_toolkit = create_weather_toolkit()
        rapidapi_weather_toolkit = create_rapidapi_weather_toolkit()
        
        # 创建智能代理，整合多个API服务
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
            tools=[weather_toolkit, rapidapi_weather_toolkit],  # 使用多个工具集
            max_tool_calls=3,
            # llm_config=LLMConfig(model="gpt-3.5-turbo")  # 需要配置LLM
        )
        
        return agent
        
    except ImportError as e:
        print(f"无法导入CustomizeAgent: {e}")
        print("请确保已正确安装EvoAgentX框架")
        return None


# 示例4：工具集的高级配置
def create_advanced_api_toolkit():
    """创建具有高级配置的API工具集"""
    
    # 带有复杂认证的API规范
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
    
    # 创建带有自定义配置的工具集
    toolkit = create_openapi_toolkit(
        schema_path_or_dict=advanced_api_spec,
        auth_config={
            "api_key": "your-api-key",
            "bearer_token": "your-bearer-token"  # 支持多种认证方式
        }
    )
    
    # 添加自定义请求头
    toolkit.common_headers.update({
        "User-Agent": "EvoAgentX-Client/1.0",
        "Accept": "application/json",
        "Cache-Control": "no-cache"
    })
    
    # 为工具添加自定义行为
    for tool in toolkit.tools:
        original_function = tool.function
        
        def create_enhanced_wrapper(func):
            def enhanced_wrapper(**kwargs):
                # 添加请求前处理
                print(f"正在调用API工具: {tool.name}")
                print(f"参数: {kwargs}")
                
                try:
                    result = func(**kwargs)
                    # 添加响应后处理
                    print(f"API调用成功，返回结果类型: {type(result)}")
                    return result
                except Exception as e:
                    print(f"API调用失败: {e}")
                    # 可以添加重试逻辑或错误处理
                    raise
                    
            return enhanced_wrapper
        
        tool.function = create_enhanced_wrapper(original_function)
    
    return toolkit


# 天气API功能演示函数
def demo_weather_functions():
    """演示三个天气API功能的详细使用"""
    print("=== 天气API三大功能演示 ===\n")
    
    try:
        # 创建工具集
        toolkit = create_rapidapi_weather_toolkit()
        print(f"工具集创建成功，包含 {len(toolkit.tools)} 个工具:")
        for tool in toolkit.tools:
            print(f"  - {tool.name}: {tool.description}")
        print()
        
        # 获取工具
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
        
        # 功能1：按城市名称查询天气
        print("【功能1】按城市名称查询天气")
        print("-" * 40)
        if city_tool:
            cities = ["new york", "beijing", "tokyo"]
            for city in cities:
                try:
                    print(f"查询 {city} 的天气...")
                    result = city_tool.function(city=city, lang="EN")
                    
                    name = result.get('name', 'Unknown')
                    country = result.get('sys', {}).get('country', 'Unknown')
                    temp = result.get('main', {}).get('temp', 'N/A')
                    humidity = result.get('main', {}).get('humidity', 'N/A')
                    weather = result.get('weather', [{}])[0].get('description', 'N/A')
                    
                    print(f"  ✓ {name}, {country}")
                    print(f"    温度: {temp}°F, 湿度: {humidity}%")
                    print(f"    天气: {weather}")
                    print()
                except Exception as e:
                    print(f"  ✗ 查询 {city} 失败: {e}")
                    print()
        
        # 功能2：按经纬度查询天气
        print("【功能2】按经纬度查询天气")
        print("-" * 40)
        if coord_tool:
            coordinates = [
                ("40.730610", "-73.935242", "纽约地区"),
                ("39.904200", "116.407396", "北京地区"),
                ("35.689487", "139.691706", "东京地区")
            ]
            
            for lat, lon, desc in coordinates:
                try:
                    print(f"查询 {desc} ({lat}, {lon}) 的天气...")
                    result = coord_tool.function(latitude=lat, longitude=lon, lang="EN")
                    
                    name = result.get('name', 'Unknown')
                    country = result.get('sys', {}).get('country', 'Unknown')
                    temp = result.get('main', {}).get('temp', 'N/A')
                    humidity = result.get('main', {}).get('humidity', 'N/A')
                    weather = result.get('weather', [{}])[0].get('description', 'N/A')
                    
                    print(f"  ✓ {name}, {country}")
                    print(f"    温度: {temp}K, 湿度: {humidity}%")
                    print(f"    天气: {weather}")
                    print()
                except Exception as e:
                    print(f"  ✗ 查询 {desc} 失败: {e}")
                    print()
        
        # 功能3：5天天气预报
        print("【功能3】5天天气预报")
        print("-" * 40)
        if forecast_tool:
            try:
                print("获取纽约地区5天天气预报...")
                result = forecast_tool.function(latitude="40.730610", longitude="-73.935242", lang="EN")
                
                city_info = result.get('city', {})
                city_name = city_info.get('name', 'Unknown')
                country = city_info.get('country', 'Unknown')
                cnt = result.get('cnt', 0)
                
                print(f"  ✓ {city_name}, {country} - 共 {cnt} 条预报数据")
                
                forecasts = result.get('list', [])
                if forecasts:
                    print("  未来几天天气预报:")
                    for i, forecast in enumerate(forecasts):  # 显示前10条
                        dt_txt = forecast.get('dt_txt', 'Unknown')
                        temp = forecast.get('main', {}).get('temp', 'N/A')
                        weather_desc = forecast.get('weather', [{}])[0].get('description', 'N/A')
                        humidity = forecast.get('main', {}).get('humidity', 'N/A')
                        
                        print(f"    {dt_txt}: {temp}K, {weather_desc}, 湿度{humidity}%")
                        
                        if 'rain' in forecast:
                            rain = forecast['rain'].get('3h', 0)
                            if rain > 0:
                                print(f"      降雨量: {rain}mm")
                print()
            except Exception as e:
                print(f"  ✗ 获取5天预报失败: {e}")
                print()
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
    
    print("=== 演示完成 ===")


# 使用示例
def main():
    """主函数，演示各种API集成"""
    
    print("=== API转换器真实世界示例 ===\n")
    
    # # 1. 创建天气工具集
    # print("1. 创建天气API工具集...")
    # try:
    #     weather_toolkit = create_weather_toolkit()
    #     print(f"   天气工具集创建成功: {weather_toolkit.name}")
    #     print(f"   包含工具: {[tool.name for tool in weather_toolkit.tools]}")
        
    #     # 演示天气工具使用（需要有效的API密钥）
    #     if os.getenv("OPENWEATHER_API_KEY"):
    #         weather_tool = weather_toolkit.get_tool("getCurrentWeather")
    #         # result = weather_tool(q="Beijing,CN", units="metric")
    #         # print(f"   北京天气: {result}")
    #     else:
    #         print("   提示: 设置OPENWEATHER_API_KEY环境变量以测试实际API调用")
            
    # except Exception as e:
    #     print(f"   创建天气工具集时出错: {e}")
    
    # print()
    
    # 2. 创建RapidAPI天气工具集（包含三个功能）
    print("2. 创建RapidAPI天气API工具集（包含三个功能）...")
    try:
        rapidapi_weather_toolkit = create_rapidapi_weather_toolkit()
        print(f"   RapidAPI天气工具集创建成功: {rapidapi_weather_toolkit.name}")
        print(f"   包含工具: {[tool.name for tool in rapidapi_weather_toolkit.tools]}")
        print(f"   RapidAPI头配置: {rapidapi_weather_toolkit.common_headers}")
        
        # 获取所有三个工具
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
        
        print("\n   === 功能1：按城市名称查询天气 ===")
        if city_tool:
            try:
                print("   正在查询纽约天气...")
                result = city_tool.function(city="new york", lang="EN")
                print("   ✓ 纽约天气查询成功!")
                print(f"   城市: {result.get('name', 'N/A')}")
                if 'main' in result:
                    temp_f = result['main'].get('temp', 'N/A')
                    humidity = result['main'].get('humidity', 'N/A')
                    print(f"   温度: {temp_f}°F")
                    print(f"   湿度: {humidity}%")
                if 'weather' in result and result['weather']:
                    weather_desc = result['weather'][0].get('description', 'N/A')
                    print(f"   天气状况: {weather_desc}")
                if 'coord' in result:
                    lon = result['coord'].get('lon', 'N/A')
                    lat = result['coord'].get('lat', 'N/A')
                    print(f"   经纬度: {lat}, {lon}")
            except Exception as api_error:
                print(f"   ✗ 城市天气查询失败: {api_error}")
        
        print("\n   === 功能2：按经纬度查询天气 ===")
        if latlon_tool:
            try:
                print("   正在查询经纬度(40.730610, -73.935242)天气...")
                result = latlon_tool.function(latitude="40.730610", longitude="-73.935242", lang="EN")
                print("   ✓ 经纬度天气查询成功!")
                print(f"   地点: {result.get('name', 'N/A')}")
                if 'main' in result:
                    temp = result['main'].get('temp', 'N/A')
                    humidity = result['main'].get('humidity', 'N/A')
                    print(f"   温度: {temp}K")
                    print(f"   湿度: {humidity}%")
                if 'weather' in result and result['weather']:
                    weather_desc = result['weather'][0].get('description', 'N/A')
                    print(f"   天气状况: {weather_desc}")
            except Exception as api_error:
                print(f"   ✗ 经纬度天气查询失败: {api_error}")
        
        print("\n   === 功能3：5天天气预报 ===")
        if forecast_tool:
            try:
                print("   正在获取5天天气预报...")
                result = forecast_tool.function(latitude="40.730610", longitude="-73.935242", lang="EN")
                print("   ✓ 5天天气预报获取成功!")
                print(f"   预报条数: {result.get('cnt', 'N/A')}")
                if 'city' in result:
                    city_name = result['city'].get('name', 'N/A')
                    print(f"   城市: {city_name}")
                if 'list' in result and result['list']:
                    print("   前3天预报:")
                    for i, forecast in enumerate(result['list'][:3]):
                        dt_txt = forecast.get('dt_txt', 'N/A')
                        if 'main' in forecast:
                            temp = forecast['main'].get('temp', 'N/A')
                            print(f"     {dt_txt}: {temp}K")
                        if 'weather' in forecast and forecast['weather']:
                            desc = forecast['weather'][0].get('description', 'N/A')
                            print(f"       天气: {desc}")
            except Exception as api_error:
                print(f"   ✗ 5天天气预报获取失败: {api_error}")
            
    except Exception as e:
        print(f"   创建RapidAPI天气工具集时出错: {e}")
    
    print()
    
    # 4. 创建多服务代理
    # print("3. 创建多服务智能代理...")
    # try:
    #     agent = create_multi_service_agent()
    #     if agent:
    #         print(f"   多服务代理创建成功: {agent.name}")
    #         print(f"   代理描述: {agent.description}")
            
    #         # 演示代理使用
    #         # result = agent(inputs={"user_request": "What's the weather like in Tokyo?"})
    #         # print(f"   代理响应: {result}")
    #     else:
    #         print("   多服务代理创建失败（可能缺少依赖）")
            
    # except Exception as e:
    #     print(f"   创建多服务代理时出错: {e}")
    
    # print()
    
    # # 4. 创建高级配置工具集
    # print("4. 创建高级配置API工具集...")
    # try:
    #     advanced_toolkit = create_advanced_api_toolkit()
    #     print(f"   高级工具集创建成功: {advanced_toolkit.name}")
    #     print(f"   自定义请求头: {advanced_toolkit.common_headers}")
        
    # except Exception as e:
    #     print(f"   创建高级工具集时出错: {e}")
    
    print("\n=== 基础示例完成 ===")
    print("\n现在演示完整的天气API三大功能...")
    print("=" * 60)
    
    # 演示三个天气API功能
    demo_weather_functions()


if __name__ == "__main__":
    main()
