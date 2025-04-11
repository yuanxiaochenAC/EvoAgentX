# EvoAgentX API 文档

## 认证

所有API端点都需要使用Bearer token进行认证。要获取token，请使用登录端点：

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@clayx.ai&password=adminpassword"
```

响应：
```json
{
    "access_token": "your_access_token",
    "token_type": "bearer"
}
```

在所有后续请求中，需要在Authorization头中包含访问令牌：
```
Authorization: Bearer your_access_token
```

## Agent API

### 创建Agent

创建一个新的agent，并配置其参数。

**功能说明：**
- 创建新的agent实例
- 配置LLM模型参数
- 设置agent的系统提示词
- 添加标签和描述

**示例：**
```bash
curl -X POST "http://localhost:8000/api/v1/agents" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_access_token" \
  -d '{
    "name": "MyAgent",
    "description": "A helpful agent for various tasks",
    "config": {
        "llm_type": "OpenAILLM",
        "model": "gpt-3.5-turbo",
        "openai_key": "your-openai-key",
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 0.9,
        "output_response": true,
        "prompt": "You are a helpful assistant that can help with a variety of tasks."
    },
    "runtime_params": {},
    "tags": ["test", "agent_generation"]
  }'
```

响应 (200/201):
```json
{
    "_id": "agent_id",
    "name": "MyAgent",
    "description": "A helpful agent for various tasks",
    "config": {
        "llm_type": "OpenAILLM",
        "model": "gpt-3.5-turbo",
        "openai_key": "your-openai-key",
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 0.9,
        "output_response": true,
        "prompt": "You are a helpful assistant that can help with a variety of tasks."
    },
    "runtime_params": {},
    "tags": ["test", "agent_generation"],
    "created_by": "user_id",
    "created_at": "2024-03-21T10:00:00Z",
    "updated_at": "2024-03-21T10:00:00Z",
    "status": "CREATED"
}
```

### 获取Agent

通过ID获取特定agent的详细信息。

**功能说明：**
- 获取agent的完整配置信息
- 查看agent的创建和更新时间
- 获取agent的当前状态

**示例：**
```bash
curl -X GET "http://localhost:8000/api/v1/agents/agent_id" \
  -H "Authorization: Bearer your_access_token"
```

响应 (200):
```json
{
    "_id": "agent_id",
    "name": "MyAgent",
    "description": "A helpful agent for various tasks",
    "config": {
        "llm_type": "OpenAILLM",
        "model": "gpt-3.5-turbo",
        "openai_key": "your-openai-key",
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 0.9,
        "output_response": true,
        "prompt": "You are a helpful assistant that can help with a variety of tasks."
    },
    "runtime_params": {},
    "tags": ["test", "agent_generation"],
    "created_by": "user_id",
    "created_at": "2024-03-21T10:00:00Z",
    "updated_at": "2024-03-21T10:00:00Z",
    "status": "CREATED"
}
```

### 更新Agent

更新现有agent的配置。

**功能说明：**
- 修改agent的名称和描述
- 更新LLM配置参数
- 更新系统提示词
- 修改标签

**示例：**
```bash
curl -X PUT "http://localhost:8000/api/v1/agents/agent_id" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_access_token" \
  -d '{
    "name": "UpdatedAgentName",
    "description": "Updated description",
    "config": {
        "temperature": 0.8,
        "prompt": "You are an updated assistant with new capabilities."
    },
    "tags": ["test", "updated"]
  }'
```

响应 (200):
```json
{
    "_id": "agent_id",
    "name": "UpdatedAgentName",
    "description": "Updated description",
    "config": {
        "llm_type": "OpenAILLM",
        "model": "gpt-3.5-turbo",
        "openai_key": "your-openai-key",
        "temperature": 0.8,
        "max_tokens": 150,
        "top_p": 0.9,
        "output_response": true,
        "prompt": "You are an updated assistant with new capabilities."
    },
    "runtime_params": {},
    "tags": ["test", "updated"],
    "created_by": "user_id",
    "created_at": "2024-03-21T10:00:00Z",
    "updated_at": "2024-03-21T11:00:00Z",
    "status": "CREATED"
}
```

### 删除Agent

删除指定的agent（仅管理员可用）。

**功能说明：**
- 永久删除agent及其配置
- 检查agent是否被工作流使用
- 如果agent正在使用中，将阻止删除

**示例：**
```bash
curl -X DELETE "http://localhost:8000/api/v1/agents/agent_id" \
  -H "Authorization: Bearer your_access_token"
```

响应 (204): 无内容

### 列出Agents

获取所有agent的列表，支持分页和搜索。

**功能说明：**
- 获取所有agent的列表
- 支持分页查询
- 支持按标签、状态、日期范围搜索
- 支持文本搜索

**示例：**
```bash
curl -X GET "http://localhost:8000/api/v1/agents?skip=0&limit=10&query=search_term&tags=tag1,tag2&status=CREATED&start_date=2024-03-01&end_date=2024-03-21" \
  -H "Authorization: Bearer your_access_token"
```

响应 (200):
```json
[
    {
        "_id": "agent_id_1",
        "name": "Agent1",
        "description": "First agent",
        "config": {
            "llm_type": "OpenAILLM",
            "model": "gpt-3.5-turbo",
            "openai_key": "your-openai-key",
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9,
            "output_response": true,
            "prompt": "You are a helpful assistant."
        },
        "runtime_params": {},
        "tags": ["test"],
        "created_by": "user_id",
        "created_at": "2024-03-21T10:00:00Z",
        "updated_at": "2024-03-21T10:00:00Z",
        "status": "CREATED"
    }
]
```

### 查询Agent

向agent发送查询并获取响应。

**功能说明：**
- 发送用户查询到agent
- 支持对话历史记录
- 获取agent的响应

**示例：**
```bash
curl -X POST "http://localhost:8000/api/v1/agents/agent_id/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_access_token" \
  -d '{
    "prompt": "What is the capital of France?",
    "history": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin."}
    ]
  }'
```

响应 (200):
```json
{
    "response": "The capital of France is Paris."
}
```

### 备份Agent

创建单个agent的备份文件。

**功能说明：**
- 将agent的当前配置保存到文件
- 自动创建备份目录（如果不存在）
- 返回备份文件的路径信息

**示例：**
```bash
curl -X POST "http://localhost:8000/api/v1/agents/agent_id/backup" \
  -H "Authorization: Bearer your_access_token" \
  --data-urlencode "backup_path=backups/my_agent_backup.json"
```

响应 (200):
```json
{
    "success": true,
    "message": "Agent backup saved to backups/my_agent_backup.json",
    "agent_id": "agent_id",
    "backup_path": "backups/my_agent_backup.json",
    "agent_name": "MyAgent"
}
```

### 还原Agent

从备份文件还原agent。

**功能说明：**
- 从备份文件创建新的agent
- 自动生成新的唯一名称
- 保留原始agent的配置和参数

**示例：**
```bash
curl -X POST "http://localhost:8000/api/v1/agents/restore" \
  -H "Authorization: Bearer your_access_token" \
  --data-urlencode "backup_path=backups/my_agent_backup.json"
```

响应 (200):
```json
{
    "success": true,
    "message": "Agent restored from backups/my_agent_backup.json with new name MyAgent_a1b2c3d4",
    "agent_id": "new_agent_id",
    "backup_path": "backups/my_agent_backup.json",
    "agent_name": "MyAgent_a1b2c3d4"
}
```

### 列出Agent备份

获取特定agent的所有备份文件列表。

**功能说明：**
- 列出指定agent的所有备份文件
- 显示备份文件的创建时间和大小
- 按创建时间排序（最新的优先）

**示例：**
```bash
curl -X GET "http://localhost:8000/api/v1/agents/agent_id/backups" \
  -H "Authorization: Bearer your_access_token" \
  --data-urlencode "backup_dir=backups"
```

响应 (200):
```json
[
    {
        "path": "backups/MyAgent_v1.json",
        "created_at": "2024-03-21T15:30:00Z",
        "size": 1024
    },
    {
        "path": "backups/MyAgent_v2.json",
        "created_at": "2024-03-22T10:15:00Z",
        "size": 1056
    }
]
```

### 备份所有Agent

将系统中的所有agent备份到指定目录。

**功能说明：**
- 自动备份所有现有agent
- 创建备份目录（如果不存在）
- 提供详细的备份结果报告

**示例：**
```bash
curl -X POST "http://localhost:8000/api/v1/agents/backup-all" \
  -H "Authorization: Bearer your_access_token" \
  --data-urlencode "backup_dir=backups/all_agents"
```

响应 (200):
```json
{
    "success": true,
    "total": 3,
    "successful": 3,
    "failed": 0,
    "backup_dir": "backups/all_agents",
    "results": [
        {
            "success": true,
            "message": "Agent backup saved to backups/all_agents/Agent1_backup.json",
            "agent_id": "agent_id_1",
            "backup_path": "backups/all_agents/Agent1_backup.json",
            "agent_name": "Agent1"
        },
        // 更多备份结果...
    ]
}
```

### 批量备份Agent

备份指定的多个agent。

**功能说明：**
- 根据提供的agent ID列表执行备份
- 仅备份在AgentManager中存在的agent
- 提供详细的成功/失败报告

**示例：**
```bash
curl -X POST "http://localhost:8000/api/v1/agents/backup-batch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_access_token" \
  -d '{
    "agent_ids": ["agent_id_1", "agent_id_2"]
  }' \
  --data-urlencode "backup_dir=backups/selected_agents"
```

响应 (200):
```json
{
    "success": true,
    "total": 2,
    "successful": 2,
    "failed": 0,
    "backup_dir": "backups/selected_agents",
    "results": [
        {
            "success": true,
            "message": "Agent backup saved to backups/selected_agents/Agent1_backup.json",
            "agent_id": "agent_id_1",
            "backup_path": "backups/selected_agents/Agent1_backup.json",
            "agent_name": "Agent1"
        },
        // 更多备份结果...
    ]
}
```

### 批量还原Agent

从多个备份文件还原agent。

**功能说明：**
- 从指定的备份文件列表还原多个agent
- 为每个还原的agent生成唯一名称
- 提供详细的还原结果报告

**示例：**
```bash
curl -X POST "http://localhost:8000/api/v1/agents/restore-batch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_access_token" \
  -d '{
    "backup_files": [
      "backups/Agent1_backup.json",
      "backups/Agent2_backup.json"
    ]
  }'
```

响应 (200):
```json
{
    "success": true,
    "total": 2,
    "successful": 2,
    "failed": 0,
    "results": [
        {
            "success": true,
            "message": "Agent restored from backups/Agent1_backup.json with new name Agent1_a1b2c3d4",
            "agent_id": "new_agent_id_1",
            "backup_path": "backups/Agent1_backup.json",
            "agent_name": "Agent1_a1b2c3d4"
        },
        // 更多还原结果...
    ]
}
```

### 还原所有Agent

从指定目录还原所有agent备份。

**功能说明：**
- 从备份目录中的所有JSON文件还原agent
- 自动扫描目录中的所有备份文件
- 提供详细的还原结果报告

**示例：**
```bash
curl -X POST "http://localhost:8000/api/v1/agents/restore-all" \
  -H "Authorization: Bearer your_access_token" \
  --data-urlencode "backup_dir=backups/all_agents"
```

响应 (200):
```json
{
    "success": true,
    "total": 3,
    "successful": 3,
    "failed": 0,
    "results": [
        {
            "success": true,
            "message": "Agent restored from backups/all_agents/Agent1_backup.json with new name Agent1_a1b2c3d4",
            "agent_id": "new_agent_id_1",
            "backup_path": "backups/all_agents/Agent1_backup.json",
            "agent_name": "Agent1_a1b2c3d4"
        },
        // 更多还原结果...
    ]
}
```

## 错误响应

所有端点可能返回以下错误响应：

- 400 Bad Request: 无效的输入数据或配置
- 401 Unauthorized: 缺少或无效的认证令牌
- 403 Forbidden: 权限不足
- 404 Not Found: 资源未找到
- 500 Internal Server Error: 服务器端错误

错误响应示例：
```json
{
    "detail": "Error message describing what went wrong"
}
```

## 注意事项

1. 所有时间戳均使用ISO 8601格式
2. Agent名称必须唯一
3. 如果agent正在被工作流使用，则无法删除
4. 配置中的`openai_key`应保持安全，不应在日志或响应中暴露
5. 删除agent需要管理员权限 