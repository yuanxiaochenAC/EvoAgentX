import json
import requests
import inspect
from typing import Dict, List, Optional, Any, Union, Callable
from abc import ABC, abstractmethod
from functools import partial

from .tool import Tool, Toolkit
from ..core.logging import logger
from ..core.module import BaseModule


class APITool(Tool):
    """
    API工具包装器，将单个API端点包装为Tool (转为英文)
    
    Attributes:
        name: 工具名称
        description: 工具描述
        inputs: 输入参数规范
        required: 必需参数列表
        endpoint_config: API端点配置
        auth_config: 认证配置
        function: 实际执行函数
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        inputs: Dict[str, Dict[str, Any]],
        required: Optional[List[str]] = None,
        endpoint_config: Dict[str, Any] = None,
        auth_config: Dict[str, Any] = None,
        function: Callable = None
    ):
        super().__init__(name=name, description=description, inputs=inputs, required=required)
        self.endpoint_config = endpoint_config or {}
        self.auth_config = auth_config or {}
        self.function = function
    
    @property
    def __name__(self):
        return self.name
    
    def __call__(self, **kwargs):
        """执行API调用"""
        if not self.function:
            raise ValueError("Function not set for APITool")
        
        try:
            result = self.function(**kwargs)
            return self._process_result(result)
        except Exception as e:
            logger.error(f"Error calling API tool {self.name}: {str(e)}")
            raise
    
    def _process_result(self, result: Any) -> Any:
        """处理API返回结果"""
        if isinstance(result, requests.Response):
            try:
                return result.json()
            except:
                return result.text
        return result
    
    @classmethod
    def validate_attributes(cls):
        """验证属性"""
        # APITool的属性在实例化时设置，跳过类级别的属性验证
        # 只有在子类中定义了类级别属性时才进行验证
        if cls.__name__ == 'APITool':
            return
        
        # 继承父类验证，但放宽对__call__方法的要求
        required_attributes = {
            "name": str,
            "description": str,
            "inputs": dict
        }
        
        for attr, attr_type in required_attributes.items():
            if not hasattr(cls, attr):
                raise ValueError(f"Attribute {attr} is required")
            if not isinstance(getattr(cls, attr), attr_type):
                raise ValueError(f"Attribute {attr} must be of type {attr_type}")

        if hasattr(cls, 'required') and cls.required:
            for required_input in cls.required:
                if required_input not in cls.inputs:
                    raise ValueError(f"Required input '{required_input}' is not found in inputs")


class APIToolkit(Toolkit):
    """
    API工具集合，表示一个API服务的所有端点
    
    Attributes:
        name: 服务名称
        tools: API工具列表
        base_url: 基础URL
        auth_config: 认证配置
        common_headers: 通用请求头
    """
    
    def __init__(
        self,
        name: str,
        tools: List[APITool],
        base_url: str = "",
        auth_config: Dict[str, Any] = None,
        common_headers: Dict[str, str] = None
    ):
        super().__init__(name=name, tools=tools)
        self.base_url = base_url
        self.auth_config = auth_config or {}
        self.common_headers = common_headers or {}
    
    def add_auth_to_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """为请求头添加认证信息"""
        headers = headers.copy()
        headers.update(self.common_headers)
        
        # 处理不同类型的认证
        if "api_key" in self.auth_config:
            key_name = self.auth_config.get("key_name", "X-API-Key")
            headers[key_name] = self.auth_config["api_key"]
        
        if "bearer_token" in self.auth_config:
            headers["Authorization"] = f"Bearer {self.auth_config['bearer_token']}"
        
        return headers


class BaseAPIConverter(BaseModule, ABC):
    """
    基础API转换器抽象类
    
    负责将API规范转换为APIToolkit
    """
    
    def __init__(
        self,
        input_schema: Union[str, Dict[str, Any]],
        description: str = "",
        auth_config: Dict[str, Any] = None
    ):
        """
        初始化API转换器
        
        Args:
            input_schema: API规范，可以是文件路径或字典
            description: 服务描述
            auth_config: 认证配置
        """
        super().__init__()
        self.input_schema = self._load_schema(input_schema)
        self.description = description
        self.auth_config = auth_config or {}
    
    def _load_schema(self, schema: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """加载API规范"""
        if isinstance(schema, str):
            # 如果是文件路径
            try:
                with open(schema, 'r', encoding='utf-8') as f:
                    if schema.endswith('.json'):
                        return json.load(f)
                    elif schema.endswith(('.yaml', '.yml')):
                        import yaml
                        return yaml.safe_load(f)
                    else:
                        # 尝试JSON解析
                        content = f.read()
                        return json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load schema from {schema}: {e}")
                raise
        elif isinstance(schema, dict):
            return schema
        else:
            raise ValueError("input_schema must be a file path or dictionary")
    
    @abstractmethod
    def convert_to_toolkit(self) -> APIToolkit:
        """
        将API规范转换为APIToolkit
        
        Returns:
            APIToolkit: 转换后的工具集
        """
        pass
    
    @abstractmethod
    def _create_api_function(self, endpoint_config: Dict[str, Any]) -> Callable:
        """
        为单个API端点创建执行函数
        
        Args:
            endpoint_config: 端点配置
            
        Returns:
            Callable: API执行函数
        """
        pass
    
    def _extract_parameters(self, endpoint_config: Dict[str, Any]) -> tuple:
        """
        从端点配置中提取参数信息
        
        Args:
            endpoint_config: 端点配置
            
        Returns:
            tuple: (inputs, required) 参数规范和必需参数列表
        """
        inputs = {}
        required = []
        
        # 这里需要根据具体的API规范格式来实现
        # 默认实现，子类可以重写
        parameters = endpoint_config.get("parameters", [])
        
        for param in parameters:
            param_name = param.get("name", "")
            param_type = param.get("type", "string")
            param_desc = param.get("description", "")
            is_required = param.get("required", False)
            
            inputs[param_name] = {
                "type": param_type,
                "description": param_desc
            }
            
            if is_required:
                required.append(param_name)
        
        return inputs, required


class OpenAPIConverter(BaseAPIConverter):
    """
    OpenAPI (Swagger) 规范转换器
    """
    
    def convert_to_toolkit(self) -> APIToolkit:
        """将OpenAPI规范转换为APIToolkit"""
        service_name = self.input_schema.get("info", {}).get("title", "API Service")
        base_url = self._get_base_url()
        
        tools = []
        paths = self.input_schema.get("paths", {})
        
        for path, methods in paths.items():
            for method, operation in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    tool = self._create_tool_from_operation(path, method, operation, base_url)
                    if tool:
                        tools.append(tool)
        
        return APIToolkit(
            name=service_name,
            tools=tools,
            base_url=base_url,
            auth_config=self.auth_config,
            common_headers={"Content-Type": "application/json"}
        )
    
    def _get_base_url(self) -> str:
        """从OpenAPI规范中获取基础URL"""
        servers = self.input_schema.get("servers", [])
        if servers:
            return servers[0].get("url", "")
        
        # 旧版本的host和basePath
        host = self.input_schema.get("host", "")
        base_path = self.input_schema.get("basePath", "")
        schemes = self.input_schema.get("schemes", ["https"])
        
        if host:
            return f"{schemes[0]}://{host}{base_path}"
        
        return ""
    
    def _create_tool_from_operation(
        self, 
        path: str, 
        method: str, 
        operation: Dict[str, Any], 
        base_url: str
    ) -> Optional[APITool]:
        """从OpenAPI操作创建工具"""
        try:
            # 生成工具名称
            operation_id = operation.get("operationId")
            if not operation_id:
                # 如果没有operationId，根据路径和方法生成
                clean_path = path.replace("/", "_").replace("{", "").replace("}", "").strip("_")
                operation_id = f"{method.lower()}_{clean_path}"
            
            # 提取参数
            inputs, required = self._extract_openapi_parameters(operation)
            
            # 创建API执行函数
            api_function = self._create_api_function({
                "url": base_url + path,
                "method": method.upper(),
                "operation": operation
            })
            
            return APITool(
                name=operation_id,
                description=operation.get("summary", operation.get("description", "")),
                inputs=inputs,
                required=required,
                endpoint_config={
                    "url": base_url + path,
                    "method": method.upper(),
                    "operation": operation
                },
                auth_config=self.auth_config,
                function=api_function
            )
        except Exception as e:
            logger.warning(f"Failed to create tool for {method.upper()} {path}: {e}")
            return None
    
    def _extract_openapi_parameters(self, operation: Dict[str, Any]) -> tuple:
        """从OpenAPI操作中提取参数"""
        inputs = {}
        required = []
        
        # 处理parameters
        parameters = operation.get("parameters", [])
        for param in parameters:
            param_name = param.get("name", "")
            param_schema = param.get("schema", {})
            param_type = param_schema.get("type", "string")
            
            inputs[param_name] = {
                "type": param_type,
                "description": param.get("description", "")
            }
            
            if param.get("required", False):
                required.append(param_name)
        
        # 处理requestBody
        request_body = operation.get("requestBody", {})
        if request_body:
            content = request_body.get("content", {})
            for media_type, media_schema in content.items():
                if "application/json" in media_type:
                    schema = media_schema.get("schema", {})
                    properties = schema.get("properties", {})
                    
                    for prop_name, prop_schema in properties.items():
                        inputs[prop_name] = {
                            "type": prop_schema.get("type", "string"),
                            "description": prop_schema.get("description", "")
                        }
                        
                        if prop_name in schema.get("required", []):
                            required.append(prop_name)
        
        return inputs, required
    
    def _create_api_function(self, endpoint_config: Dict[str, Any]) -> Callable:
        """创建OpenAPI执行函数"""
        url = endpoint_config["url"]
        method = endpoint_config["method"]
        operation = endpoint_config["operation"]
        
        def api_call(**kwargs):
            # 分离路径参数、查询参数和请求体
            path_params = {}
            query_params = {}
            body_data = {}
            
            parameters = operation.get("parameters", [])
            param_locations = {param["name"]: param.get("in", "query") for param in parameters}
            
            for key, value in kwargs.items():
                if value is None:
                    continue
                    
                location = param_locations.get(key, "body")
                if location == "path":
                    path_params[key] = value
                elif location == "query":
                    query_params[key] = value
                else:
                    body_data[key] = value
            
            # 替换路径参数
            final_url = url
            for param_name, param_value in path_params.items():
                final_url = final_url.replace(f"{{{param_name}}}", str(param_value))
            
            # 准备请求
            headers = {"Content-Type": "application/json"}
            if hasattr(self, 'auth_config') and self.auth_config:
                if "api_key" in self.auth_config:
                    key_name = self.auth_config.get("key_name", "X-API-Key")
                    headers[key_name] = self.auth_config["api_key"]
            
            # 发送请求
            try:
                if method in ["GET", "DELETE"]:
                    response = requests.request(
                        method=method,
                        url=final_url,
                        params=query_params,
                        headers=headers,
                        timeout=30
                    )
                else:
                    response = requests.request(
                        method=method,
                        url=final_url,
                        params=query_params,
                        json=body_data if body_data else None,
                        headers=headers,
                        timeout=30
                    )
                
                response.raise_for_status()
                
                try:
                    return response.json()
                except:
                    return response.text
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                raise
        
        # 设置函数名称以便调试
        api_call.__name__ = f"api_call_{method.lower()}"
        return api_call


class RapidAPIConverter(OpenAPIConverter):
    """
    RapidAPI专用转换器
    继承自OpenAPIConverter，添加RapidAPI特定的认证和配置
    """
    
    def __init__(
        self,
        input_schema: Union[str, Dict[str, Any]],
        description: str = "",
        rapidapi_key: str = "",
        rapidapi_host: str = "",
        **kwargs
    ):
        """
        初始化RapidAPI转换器
        
        Args:
            input_schema: API规范
            description: 服务描述
            rapidapi_key: RapidAPI密钥
            rapidapi_host: RapidAPI主机
        """
        # 设置RapidAPI特定的认证配置
        auth_config = {
            "api_key": rapidapi_key,
            "key_name": "X-RapidAPI-Key",
            "rapidapi_host": rapidapi_host
        }
        
        super().__init__(
            input_schema=input_schema,
            description=description,
            auth_config=auth_config,
            **kwargs
        )
    
    def convert_to_toolkit(self) -> APIToolkit:
        """转换为RapidAPI工具集"""
        toolkit = super().convert_to_toolkit()
        
        # 添加RapidAPI特定的通用头
        rapidapi_headers = {
            "X-RapidAPI-Key": self.auth_config.get("api_key", ""),
            "X-RapidAPI-Host": self.auth_config.get("rapidapi_host", "")
        }
        
        toolkit.common_headers.update(rapidapi_headers)
        
        return toolkit
    
    def _create_api_function(self, endpoint_config: Dict[str, Any]) -> Callable:
        """创建RapidAPI执行函数"""
        url = endpoint_config["url"]
        method = endpoint_config["method"]
        operation = endpoint_config["operation"]
        
        def rapidapi_call(**kwargs):
            # 分离参数
            path_params = {}
            query_params = {}
            body_data = {}
            
            parameters = operation.get("parameters", [])
            param_locations = {param["name"]: param.get("in", "query") for param in parameters}
            
            for key, value in kwargs.items():
                if value is None:
                    continue
                    
                location = param_locations.get(key, "body")
                if location == "path":
                    path_params[key] = value
                elif location == "query":
                    query_params[key] = value
                else:
                    body_data[key] = value
            
            # 替换路径参数
            final_url = url
            for param_name, param_value in path_params.items():
                final_url = final_url.replace(f"{{{param_name}}}", str(param_value))
            
            # 准备RapidAPI请求头
            headers = {
                "Content-Type": "application/json",
                "X-RapidAPI-Key": self.auth_config.get("api_key", ""),
                "X-RapidAPI-Host": self.auth_config.get("rapidapi_host", "")
            }
            
            # 发送请求
            try:
                if method in ["GET", "DELETE"]:
                    response = requests.request(
                        method=method,
                        url=final_url,
                        params=query_params,
                        headers=headers,
                        timeout=30
                    )
                else:
                    response = requests.request(
                        method=method,
                        url=final_url,
                        params=query_params,
                        json=body_data if body_data else None,
                        headers=headers,
                        timeout=30
                    )
                
                response.raise_for_status()
                
                try:
                    return response.json()
                except:
                    return response.text
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"RapidAPI request failed: {e}")
                raise
        
        rapidapi_call.__name__ = f"rapidapi_call_{method.lower()}"
        return rapidapi_call


def create_openapi_toolkit(
    schema_path_or_dict: Union[str, Dict[str, Any]],
    service_name: str = None,
    auth_config: Dict[str, Any] = None
) -> APIToolkit:
    """
    便捷函数：从OpenAPI规范创建APIToolkit
    
    Args:
        schema_path_or_dict: OpenAPI规范文件路径或字典
        service_name: 服务名称（可选，会从规范中提取）
        auth_config: 认证配置
    
    Returns:
        APIToolkit: 创建的工具集
    """
    converter = OpenAPIConverter(
        input_schema=schema_path_or_dict,
        description=service_name or "",
        auth_config=auth_config
    )
    return converter.convert_to_toolkit()


def create_rapidapi_toolkit(
    schema_path_or_dict: Union[str, Dict[str, Any]],
    rapidapi_key: str,
    rapidapi_host: str,
    service_name: str = None
) -> APIToolkit:
    """
    便捷函数：创建RapidAPI工具集
    
    Args:
        schema_path_or_dict: API规范文件路径或字典
        rapidapi_key: RapidAPI密钥
        rapidapi_host: RapidAPI主机
        service_name: 服务名称（可选）
    
    Returns:
        APIToolkit: 创建的RapidAPI工具集
    """
    converter = RapidAPIConverter(
        input_schema=schema_path_or_dict,
        description=service_name or "",
        rapidapi_key=rapidapi_key,
        rapidapi_host=rapidapi_host
    )
    return converter.convert_to_toolkit()
