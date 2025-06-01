import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "openai/gpt-4o", # Optional
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)

# 提取返回的 JSON 数据
result = response.json()

# 打印整个返回（用于调试）
print(json.dumps(result, indent=2))

# ✅ 提取模型的回复内容（第一个 message）
#message = result['choices'][0]['message']['content']
# print("Model reply:", message)
