# API转换器使用指南

API转换器是EvoAgentX框架中的一个强大工具，用于将各种API规范（如OpenAPI/Swagger、RapidAPI等）自动转换为可在智能代理中使用的工具集。

## 核心组件

### 1. APITool
单个API端点的工具包装器，继承自`Tool`类。

**主要特性:**
- 自动处理API请求和响应
- 支持各种HTTP方法（GET, POST, PUT, DELETE, PATCH）
- 灵活的参数处理（路径参数、查询参数、请求体）
- 内置错误处理和结果转换

### 2. APIToolkit  
API服务的工具集合，继承自`Toolkit`类。

**主要特性:**
- 管理多个相关的API工具
- 统一的认证配置
- 通用请求头管理
- 服务级别的配置

### 3. BaseAPIConverter
基础API转换器抽象类，定义了转换接口。

**核心方法:**
- `convert_to_toolkit()`: 将API规范转换为APIToolkit
- `_create_api_function()`: 为单个端点创建执行函数
- `_extract_parameters()`: 提取参数信息

### 4. OpenAPIConverter
OpenAPI (Swagger) 规范转换器，继承自`BaseAPIConverter`。

**支持特性:**
- OpenAPI 3.0+ 规范
- 自动参数提取
- 路径参数替换
- 请求体处理
- 响应格式化

### 5. RapidAPIConverter
RapidAPI专用转换器，继承自`OpenAPIConverter`。

**RapidAPI特性:**
- 自动添加RapidAPI认证头
- 支持RapidAPI主机配置
- 专用的错误处理

## 使用方法

### 基本用法

#### 1. 从OpenAPI规范创建工具集

```python
from evoagentx.tools.api_converter import create_api_toolkit_from_openapi

# 从字典创建
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

toolkit = create_api_toolkit_from_openapi(
    schema_path_or_dict=openapi_spec,
    auth_config={"api_key": "your-api-key"}
)

# 从文件创建
toolkit = create_api_toolkit_from_openapi(
    schema_path_or_dict="path/to/openapi.json",
    auth_config={"api_key": "your-api-key"}
)
```

#### 2. 创建RapidAPI工具集

```python
from evoagentx.tools.api_converter import create_rapidapi_toolkit

toolkit = create_rapidapi_toolkit(
    schema_path_or_dict="path/to/rapidapi_spec.json",
    rapidapi_key="your-rapidapi-key",
    rapidapi_host="api.rapidapi.host.com"
)
```

#### 3. 在CustomizeAgent中使用

```python
from evoagentx.agents.customize_agent import CustomizeAgent
from evoagentx.tools.api_converter import create_api_toolkit_from_openapi

# 创建API工具集
api_toolkit = create_api_toolkit_from_openapi(
    schema_path_or_dict=openapi_spec,
    auth_config={"api_key": "your-api-key"}
)

# 创建使用API工具的智能代理
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
    tools=[api_toolkit]  # 使用API工具集
)

# 使用代理
result = agent(inputs={"city": "Beijing"})
```

### 高级用法

#### 1. 自定义转换器

```python
from evoagentx.tools.api_converter import BaseAPIConverter

class CustomAPIConverter(BaseAPIConverter):
    def convert_to_toolkit(self):
        # 实现自定义转换逻辑
        pass
    
    def _create_api_function(self, endpoint_config):
        # 实现自定义API函数创建
        pass
```

#### 2. 复杂认证配置

```python
# API密钥认证
auth_config = {
    "api_key": "your-api-key",
    "key_name": "X-API-Key"  # 自定义头名称
}

# Bearer Token认证
auth_config = {
    "bearer_token": "your-bearer-token"
}

# 组合认证
auth_config = {
    "api_key": "your-api-key",
    "bearer_token": "your-bearer-token"
}
```

#### 3. 自定义请求头

```python
from evoagentx.tools.api_converter import OpenAPIConverter

converter = OpenAPIConverter(
    input_schema=openapi_spec,
    auth_config=auth_config
)

toolkit = converter.convert_to_toolkit()

# 添加自定义请求头
toolkit.common_headers.update({
    "User-Agent": "MyApp/1.0",
    "Accept": "application/json"
})
```

## 支持的API规范格式

### OpenAPI/Swagger
- **版本**: OpenAPI 3.0+, Swagger 2.0
- **格式**: JSON, YAML
- **特性**: 完整的参数提取、请求体处理、响应格式化

### RapidAPI
- **基础**: OpenAPI规范
- **增强**: RapidAPI特定的认证和配置
- **特性**: 自动RapidAPI头处理

## 参数类型映射

| OpenAPI类型 | Tool输入类型 | 描述 |
|-------------|-------------|------|
| string      | string      | 字符串 |
| integer     | integer     | 整数 |
| number      | number      | 数字 |
| boolean     | boolean     | 布尔值 |
| array       | array       | 数组 |
| object      | object      | 对象 |

## 错误处理

API转换器提供多层错误处理：

1. **转换时错误**: 规范解析失败、格式不正确
2. **运行时错误**: API请求失败、网络错误
3. **响应错误**: API返回错误状态码

```python
try:
    toolkit = create_api_toolkit_from_openapi(openapi_spec)
    result = toolkit.get_tool("weather_api")(city="Beijing")
except Exception as e:
    print(f"Error: {e}")
```

## 最佳实践

### 1. API密钥管理
```python
import os

# 使用环境变量存储敏感信息
api_key = os.getenv("WEATHER_API_KEY")
auth_config = {"api_key": api_key}
```

### 2. 错误重试
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

### 3. 响应缓存
```python
from functools import lru_cache

# 对稳定的API结果进行缓存
@lru_cache(maxsize=100)
def cached_api_call(endpoint, params):
    # API调用逻辑
    pass
```

## 故障排除

### 常见问题

1. **转换失败**
   - 检查OpenAPI规范格式
   - 验证必需字段是否存在
   - 确认服务器URL配置

2. **API调用失败**
   - 验证API密钥和认证配置
   - 检查网络连接
   - 确认API端点可访问

3. **参数错误**
   - 检查参数名称和类型
   - 验证必需参数是否提供
   - 确认参数值格式正确

### 调试技巧

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 检查生成的工具
for tool in toolkit.tools:
    print(f"Tool: {tool.name}")
    print(f"Inputs: {tool.inputs}")
    print(f"Required: {tool.required}")
```

## 扩展开发

### 添加新的API规范支持

1. 继承`BaseAPIConverter`
2. 实现`convert_to_toolkit()`方法
3. 实现`_create_api_function()`方法
4. 添加特定的参数提取逻辑

### 自定义工具行为

1. 继承`APITool`类
2. 重写`__call__`方法
3. 添加自定义的结果处理逻辑

## 示例项目

查看`examples/api_converter_example.py`获取完整的使用示例，包括：
- 基本OpenAPI转换
- RapidAPI集成
- CustomizeAgent集成
- 文件加载示例

---

通过API转换器，您可以轻松地将任何遵循标准规范的API集成到EvoAgentX智能代理中，大大扩展代理的能力和应用场景。
