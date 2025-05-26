# 在 EvoAgentX 中使用工具

本教程将指导你使用 EvoAgentX 强大的工具生态系统。工具允许代理与外部世界交互、执行计算和访问信息。我们将涵盖：

1. **理解工具架构**：了解基础 Tool 类及其功能
2. **代码解释器**：使用 Python 和 Docker 解释器安全执行 Python 代码
3. **搜索工具**：使用 Wikipedia 和 Google 搜索工具访问网络信息
4. **MCP 工具**：使用模型上下文协议连接到外部服务

通过本教程，你将了解如何在自己的代理和工作流中利用这些工具。

---

## 1. 理解工具架构

EvoAgentX 工具生态系统的核心是 `Tool` 基类，它为所有工具提供了标准化接口。

```python
from evoagentx.tools.tool import Tool
```

`Tool` 类实现了三个关键方法：

- `get_tool_schemas()`：返回工具的 OpenAI 兼容函数模式
- `get_tools()`：返回工具提供的可调用函数列表
- `get_tool_descriptions()`：返回工具功能描述

EvoAgentX 中的所有工具都继承自这个基类，确保代理使用它们时有一致的接口。

### 关键概念

- **工具集成**：工具通过函数调用协议与代理无缝集成
- **模式**：每个工具都提供描述其功能、参数和输出的模式
- **模块化**：工具可以轻松添加到任何支持函数调用的代理中

---

## 2. 代码解释器

EvoAgentX 提供两种主要的代码解释器工具：

1. **PythonInterpreter**：在受控环境中执行 Python 代码
2. **DockerInterpreter**：在隔离的 Docker 容器中执行代码

### 2.1 PythonInterpreter

**PythonInterpreter 提供了一个安全的环境来执行 Python 代码，可以精细控制导入、目录访问和执行上下文。它使用沙箱方法来限制潜在的有害操作。**

#### 2.1.1 设置

```python
from evoagentx.tools.interpreter_python import PythonInterpreter

# 使用特定的允许导入和目录访问进行初始化
interpreter = PythonInterpreter(
    project_path=".",  # 默认为当前目录
    directory_names=["examples", "evoagentx"],
    allowed_imports={"os", "sys", "math", "random", "datetime"}
)
```

#### 2.1.2 可用方法

`PythonInterpreter` 提供以下可调用方法：

##### 方法 1: execute(code, language)

**描述**：在安全环境中直接执行 Python 代码。

**使用示例**：
```python
# 执行简单的代码片段
result = interpreter.execute("""
print("Hello, World!")
import math
print(f"The value of pi is: {math.pi:.4f}")
""", "python")

print(result)
```

**返回类型**：`str`

**示例返回**：
```
Hello, World!
The value of pi is: 3.1416
```

---

##### 方法 2: execute_script(file_path, language)

**描述**：在安全环境中执行 Python 脚本文件。

**使用示例**：
```python
# 执行 Python 脚本文件
script_path = "examples/hello_world.py"
script_result = interpreter.execute_script(script_path, "python")
print(script_result)
```

**返回类型**：`str`

**示例返回**：
```
Running hello_world.py...
Hello from the script file!
Script execution completed.
```

#### 2.1.3 设置提示

- **项目路径**：`project_path` 参数应指向项目的根目录，以确保正确的文件访问。默认为当前目录（"."）。

- **目录名称**：`directory_names` 列表指定项目中可以导入的目录。这对安全性很重要，可以防止未授权的访问。默认为空列表 `[]`。

- **允许的导入**：`allowed_imports` 集合限制可以在执行代码中导入的 Python 模块。默认为空列表 `[]`。
  - **重要**：如果 `allowed_imports` 设置为空列表，则不应用导入限制。
  - 指定时，只添加你认为安全的模块：

```python
# 带有导入限制的示例
interpreter = PythonInterpreter(
    project_path=os.getcwd(),
    directory_names=["examples", "evoagentx", "tests"],
    allowed_imports={
        "os", "sys", "time", "datetime", "math", "random", 
        "json", "csv", "re", "collections", "itertools"
    }
)

# 无导入限制的示例
interpreter = PythonInterpreter(
    project_path=os.getcwd(),
    directory_names=["examples", "evoagentx"],
    allowed_imports=[]  # 允许导入任何模块
)
```

---

### 2.2 DockerInterpreter

**DockerInterpreter 在隔离的 Docker 容器中执行代码，提供最大的安全性和环境隔离。它允许安全执行潜在风险的代码，具有自定义环境、依赖项和完整的资源隔离。使用此工具需要在你的机器上安装并运行 Docker。**

#### 2.2.1 设置

```python
from evoagentx.tools.interpreter_docker import DockerInterpreter

# 使用特定的 Docker 镜像初始化
interpreter = DockerInterpreter(
    image_tag="fundingsocietiesdocker/python3.9-slim",
    print_stdout=True,
    print_stderr=True,
    container_directory="/app"
)
```

#### 2.2.2 可用方法

`DockerInterpreter` 提供以下可调用方法：

##### 方法 1: execute(code, language)

**描述**：在 Docker 容器内执行代码。

**使用示例**：
```python
# 在 Docker 容器中执行 Python 代码
result = interpreter.execute("""
import platform
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.system()} {platform.release()}")
""", "python")

print(result)
```

**返回类型**：`str`

**示例返回**：
```
Python version: 3.9.16
Platform: Linux 5.15.0-1031-azure
```

---

##### 方法 2: execute_script(file_path, language)

**描述**：在 Docker 容器内执行脚本文件。

**使用示例**：
```python
# 在 Docker 中执行 Python 脚本文件
script_path = "examples/docker_test.py"
script_result = interpreter.execute_script(script_path, "python")
print(script_result)
```

**返回类型**：`str`

**示例返回**：
```
Running container with script: /app/script_12345.py
Hello from the Docker container!
Container execution completed.
```

#### 2.2.3 设置提示

- **Docker 要求**：使用此解释器前，确保 Docker 已安装并在系统上运行。

- **镜像管理**：你需要提供 `image_tag` **或** `dockerfile_path` 中的一个，不能同时提供：
  - **选项 1：使用现有镜像**
    ```python
    interpreter = DockerInterpreter(
        image_tag="python:3.9-slim",  # 使用现有的 Docker Hub 镜像
        container_directory="/app"
    )
    ```
  
  - **选项 2：从 Dockerfile 构建**
    ```python
    interpreter = DockerInterpreter(
        dockerfile_path="path/to/Dockerfile",  # 构建自定义镜像
        image_tag="my-custom-image-name",      # 构建镜像的名称
        container_directory="/app"
    )
    ```

- **文件访问**：
  - 要使本地文件在容器中可用，使用 `host_directory` 参数：
  ```python
  interpreter = DockerInterpreter(
      image_tag="python:3.9-slim",
      host_directory="/path/to/local/files",
      container_directory="/app/data"
  )
  ```
  - 这将本地目录挂载到指定的容器目录，使所有文件可访问。

- **容器生命周期**：
  - Docker 容器在初始化解释器时创建，在解释器销毁时移除。
  - 对于长时间运行的会话，可以设置 `print_stdout` 和 `print_stderr` 以查看实时输出。

- **故障排除**：
  - 如果遇到权限问题，确保你的用户具有 Docker 权限。
  - 对于网络相关错误，检查 Docker 守护进程是否有适当的网络访问权限。

---

## 3. 搜索工具

EvoAgentX 提供多种搜索工具来从各种来源检索信息：

1. **SearchWiki**：搜索 Wikipedia 获取信息
2. **SearchGoogle**：使用官方 API 搜索 Google
3. **SearchGoogleFree**：无需 API 密钥即可搜索 Google

### 3.1 SearchWiki

**SearchWiki 工具从 Wikipedia 文章检索信息，提供摘要、完整内容和元数据。它提供了一种简单的方法，无需复杂的 API 设置即可将百科全书知识整合到你的代理中。**

#### 3.1.1 设置

```python
from evoagentx.tools.search_wiki import SearchWiki

# 使用自定义参数初始化
wiki_search = SearchWiki(max_sentences=3)
```

#### 3.1.2 可用方法

`SearchWiki` 提供以下可调用方法：

##### 方法: search(query)

**描述**：搜索 Wikipedia 获取与查询匹配的文章。

**使用示例**：
```python
# 搜索 Wikipedia 获取信息
results = wiki_search.search(
    query="artificial intelligence agent architecture"
)

# 处理结果
for i, result in enumerate(results.get("results", [])):
    print(f"结果 {i+1}: {result['title']}")
    print(f"摘要: {result['summary']}")
    print(f"URL: {result['url']}")
```

**返回类型**：`dict`

**示例返回**：
```python
{
    "results": [
        {
            "title": "Artificial intelligence",
            "summary": "Artificial intelligence (AI) is the intelligence of machines or software, as opposed to the intelligence of humans or animals. AI applications include advanced web search engines, recommendation systems, voice assistants...",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
        },
        {
            "title": "Intelligent agent",
            "summary": "In artificial intelligence, an intelligent agent (IA) is anything which can perceive its environment, process those perceptions, and respond in pursuit of its own goals...",
            "url": "https://en.wikipedia.org/wiki/Intelligent_agent"
        }
    ]
}
```

---

### 3.2 SearchGoogle

**SearchGoogle 工具通过 Google 的官方自定义搜索 API 实现网络搜索，提供高质量搜索结果和内容提取。它需要 API 凭证，但提供更可靠和全面的搜索功能。**

#### 3.2.1 设置

```python
from evoagentx.tools.search_google import SearchGoogle

# 使用自定义参数初始化
google_search = SearchGoogle(
    num_search_pages=3,
    max_content_words=200
)
```

#### 3.2.2 可用方法

`SearchGoogle` 提供以下可调用方法：

##### 方法: search(query)

**描述**：搜索 Google 获取与查询匹配的内容。

**使用示例**：
```python
# 搜索 Google 获取信息
results = google_search.search(
    query="evolutionary algorithms for neural networks"
)

# 处理结果
for i, result in enumerate(results.get("results", [])):
    print(f"结果 {i+1}: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"内容: {result['content'][:150]}...")
```

**返回类型**：`dict`

**示例返回**：
```python
{
    "results": [
        {
            "title": "Evolutionary Algorithms for Neural Networks - A Systematic Review",
            "url": "https://example.com/paper1",
            "content": "This paper provides a comprehensive review of evolutionary algorithms applied to neural network optimization. Key approaches include genetic algorithms, particle swarm optimization, and differential evolution..."
        },
        {
            "title": "Applying Genetic Algorithms to Neural Network Training",
            "url": "https://example.com/article2",
            "content": "Genetic algorithms offer a powerful approach to optimizing neural network architectures and weights. This article explores how evolutionary computation can overcome limitations of gradient-based methods..."
        }
    ]
}
```

#### 3.2.3 设置提示

- **API 要求**：此工具需要 Google 自定义搜索 API 凭证。在你的环境中设置它们：
  ```python
  # 在你的 .env 文件或环境变量中
  GOOGLE_API_KEY=your_google_api_key_here
  GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
  ```

- **获取凭证**：
  1. 在 [Google Cloud Console](https://console.cloud.google.com/) 创建项目
  2. 启用自定义搜索 API
  3. 创建 API 凭证
  4. 在 [https://cse.google.com/cse/](https://cse.google.com/cse/) 设置自定义搜索引擎

---

### 3.3 SearchGoogleFree

**SearchGoogleFree 工具提供网络搜索功能，无需任何 API 密钥或认证。它提供了官方 Google API 的更简单替代方案，具有适合大多数一般查询的基本搜索结果。**

#### 3.3.1 设置

```python
from evoagentx.tools.search_google_f import SearchGoogleFree

# 初始化免费 Google 搜索
google_free = SearchGoogleFree(
    num_search_pages=3,
    max_content_words=500
)
```

#### 3.3.2 可用方法

`SearchGoogleFree` 提供以下可调用方法：

##### 方法: search(query)

**描述**：无需 API 密钥即可搜索 Google 获取与查询匹配的内容。

**使用示例**：
```python
# 无需 API 密钥搜索 Google
results = google_free.search(
    query="reinforcement learning algorithms"
)

# 处理结果
for i, result in enumerate(results.get("results", [])):
    print(f"结果 {i+1}: {result['title']}")
    print(f"URL: {result['url']}")
```

**返回类型**：`dict`

**示例返回**：
```python
{
    "results": [
        {
            "title": "Introduction to Reinforcement Learning Algorithms",
            "url": "https://example.com/intro-rl",
            "snippet": "A comprehensive overview of reinforcement learning algorithms including Q-learning, SARSA, and policy gradient methods."
        },
        {
            "title": "Top 10 Reinforcement Learning Algorithms for Beginners",
            "url": "https://example.com/top-rl",
            "snippet": "Learn about the most commonly used reinforcement learning algorithms with practical examples and implementation tips."
        }
    ]
}
```

---

## 4. MCP 工具

**模型上下文协议（MCP）工具包提供了一种通过 MCP 协议连接到外部服务的标准化方法。它使代理能够访问专门的工具，如工作搜索服务、数据处理实用程序和其他 MCP 兼容的 API，而无需直接集成每个服务。**

### 4.1 MCPToolkit

#### 4.1.1 设置

```python
from evoagentx.tools.mcp import MCPToolkit

# 使用配置文件初始化
mcp_toolkit = MCPToolkit(config_path="examples/sample_mcp.config")

# 或使用配置字典初始化
config = {
    "mcpServers": {
        "hirebase": {
            "command": "uvx",
            "args": ["hirebase-mcp"],
            "env": {"HIREBASE_API_KEY": "your_api_key_here"}
        }
    }
}
mcp_toolkit = MCPToolkit(config=config)
```

#### 4.1.2 可用方法

`MCPToolkit` 提供以下可调用方法：

##### 方法 1: get_tools()

**描述**：返回从连接的 MCP 服务器获取的所有可用工具列表。

**使用示例**：
```python
# 获取所有可用的 MCP 工具
tools = mcp_toolkit.get_tools()

# 显示可用工具
for i, tool in enumerate(tools):
    print(f"工具 {i+1}: {tool.name}")
    print(f"描述: {tool.descriptions[0]}")
```

**返回类型**：`List[Tool]`

**示例返回**：
```
[MCPTool(name="HirebaseSearch", descriptions=["通过提供关键词搜索工作信息"]), 
 MCPTool(name="HirebaseAnalyze", descriptions=["分析给定技能的工作市场趋势"])]
```

---

##### 方法 2: disconnect()

**描述**：断开与所有 MCP 服务器的连接并清理资源。

**使用示例**：
```python
# 使用完 MCP 工具包后
mcp_toolkit.disconnect()
```

**返回类型**：`None`

#### 4.1.3 使用 MCP 工具

一旦从 MCPToolkit 获取了工具，你可以像使用任何其他 EvoAgentX 工具一样使用它们：

```python
# 从工具包获取所有工具
tools = mcp_toolkit.get_tools()

# 查找特定工具
hirebase_tool = None
for tool in tools:
    if "hire" in tool.name.lower() or "search" in tool.name.lower():
        hirebase_tool = tool
        break

if hirebase_tool:
    # 使用工具搜索信息
    search_query = "data scientist"
    result = hirebase_tool.tools[0](**{"query": search_query})
    
    print(f"'{search_query}' 的搜索结果：")
    print(result)
```

#### 4.1.4 设置提示

- **配置文件**：配置文件应遵循 MCP 协议的服务器配置格式：
  ```json
  {
      "mcpServers": {
          "serverName": {
              "command": "executable_command",
              "args": ["command_arguments"],
              "env": {"ENV_VAR_NAME": "value"}
          }
      }
  }
  ```

- **服务器类型**：
  - **基于命令的服务器**：使用 `command` 字段指定可执行文件
  - **基于 URL 的服务器**：使用 `url` 字段指定服务器端点

- **连接管理**：
  - 使用完 MCPToolkit 后始终调用 `disconnect()` 以释放资源
  - 使用 try-finally 块进行自动清理：
    ```python
    try:
        toolkit = MCPToolkit(config_path="config.json")
        tools = toolkit.get_tools()
        # 在这里使用工具
    finally:
        toolkit.disconnect()
    ```

- **错误处理**：
  - 如果无法连接到服务器，MCPToolkit 将记录警告消息
  - 最好在工具调用周围实现错误处理：
    ```python
    try:
        result = tool.tools[0](**{"query": "example query"})
    except Exception as e:
        print(f"调用 MCP 工具时出错：{str(e)}")
    ```

- **环境变量**：
  - API 密钥和其他敏感信息可以通过配置中的环境变量提供
  - 你也可以在运行应用程序之前在环境中设置它们

---

## 总结

在本教程中，我们探索了 EvoAgentX 中的工具生态系统：

1. **工具架构**：理解了基础 Tool 类及其标准化接口
2. **代码解释器**：学习了如何使用 Python 和 Docker 解释器安全执行 Python 代码
3. **搜索工具**：发现了如何使用 Wikipedia 和 Google 搜索工具访问网络信息
4. **MCP 工具**：学习了如何使用模型上下文协议连接到外部服务

EvoAgentX 中的工具通过提供对外部资源和计算的访问来扩展你的代理功能。通过将这些工具与代理和工作流结合，你可以构建强大的 AI 系统，能够检索信息、执行计算并与世界交互。

有关更高级的用法和自定义选项，请参考 [API 文档](../api/tools.md) 并探索仓库中的示例。 