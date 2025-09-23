import json
from evoagentx.tools.api_converter import (
    OpenAPIConverter,
    RapidAPIConverter,
    create_openapi_toolkit,
    create_rapidapi_toolkit
)

# 示例1：使用OpenAPI规范创建工具集
def example_openapi_converter():
    """使用OpenAPI规范创建工具集的示例"""
    
    # 示例OpenAPI规范（简化版）
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
    
    # 方法1：使用转换器类
    print("=== 使用OpenAPIConverter ===")
    converter = OpenAPIConverter(
        input_schema=openapi_spec,
        description="Weather service API",
        auth_config={
            "api_key": "your-api-key",
            "key_name": "X-API-Key"
        }
    )
    
    toolkit = converter.convert_to_toolkit()
    print(f"服务名称: {toolkit.name}")
    print(f"基础URL: {toolkit.base_url}")
    print(f"工具数量: {len(toolkit.tools)}")
    
    for tool in toolkit.tools:
        print(f"\n工具: {tool.name}")
        print(f"描述: {tool.description}")
        print(f"输入参数: {list(tool.inputs.keys())}")
        print(f"必需参数: {tool.required}")
    
    # 方法2：使用便捷函数
    print("\n=== 使用便捷函数 ===")
    toolkit2 = create_openapi_toolkit(
        schema_path_or_dict=openapi_spec,
        service_name="Weather Service",
        auth_config={"api_key": "your-api-key"}
    )
    print(f"便捷函数创建的工具集: {toolkit2.name}")
    
    return toolkit


# 示例2：使用RapidAPI创建工具集
def example_rapidapi_converter():
    """使用RapidAPI创建工具集的示例"""
    
    # RapidAPI的OpenAPI规范通常从RapidAPI平台获取
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
    
    print("\n=== 使用RapidAPIConverter ===")
    
    # 方法1：使用转换器类
    converter = RapidAPIConverter(
        input_schema=rapidapi_spec,
        description="Microsoft Translator API",
        rapidapi_key="your-rapidapi-key",
        rapidapi_host="microsoft-translator-text.p.rapidapi.com"
    )
    
    toolkit = converter.convert_to_toolkit()
    print(f"RapidAPI服务: {toolkit.name}")
    print(f"通用请求头: {toolkit.common_headers}")
    
    for tool in toolkit.tools:
        print(f"\n工具: {tool.name}")
        print(f"描述: {tool.description}")
        print(f"输入参数: {list(tool.inputs.keys())}")
    
    # 方法2：使用便捷函数
    toolkit2 = create_rapidapi_toolkit(
        schema_path_or_dict=rapidapi_spec,
        rapidapi_key="your-rapidapi-key",
        rapidapi_host="microsoft-translator-text.p.rapidapi.com",
        service_name="Translation Service"
    )
    
    return toolkit


# 示例3：在CustomizeAgent中使用API工具集
def example_with_customize_agent():
    """展示如何在CustomizeAgent中使用API工具集"""
    
    # 首先创建API工具集
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
    
    # 创建API工具集
    api_toolkit = create_openapi_toolkit(
        schema_path_or_dict=openapi_spec,
        auth_config={"api_key": "calc-api-key"}
    )
    
    print("\n=== 在CustomizeAgent中使用API工具 ===")
    print(f"API工具集: {api_toolkit.name}")
    print(f"可用工具: {[tool.name for tool in api_toolkit.tools]}")
    
    # 注意：实际使用时需要导入CustomizeAgent
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
    #     tools=[api_toolkit]  # 使用API工具集
    # )
    
    return api_toolkit


# 示例4：从文件加载API规范
def example_load_from_file():
    """从文件加载API规范的示例"""
    
    # 创建示例OpenAPI文件
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
    
    # 保存到文件
    spec_file = "/tmp/sample_api_spec.json"
    with open(spec_file, 'w') as f:
        json.dump(sample_spec, f, indent=2)
    
    print(f"\n=== 从文件加载API规范 ===")
    print(f"规范文件: {spec_file}")
    
    # 从文件创建工具集
    toolkit = create_openapi_toolkit(
        schema_path_or_dict=spec_file,
        service_name="Sample Service"
    )
    
    print(f"从文件创建的工具集: {toolkit.name}")
    print(f"工具: {[tool.name for tool in toolkit.tools]}")
    
    return toolkit


if __name__ == "__main__":
    print("API转换器使用示例")
    print("=" * 50)
    
    # 运行示例
    try:
        example_openapi_converter()
        example_rapidapi_converter()
        example_with_customize_agent()
        example_load_from_file()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
