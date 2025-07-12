import requests
import base64
from .tool import Tool

class OpenRouterVisionTool(Tool):
    name = "openrouter_vision"
    description = "Call OpenRouter multimodal model (e.g., gpt-4o-mini), supporting image URL, local image, base64 image, and PDF file input, and return the recognition result."
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The question or instruction for image or document analysis. Required."
        },
        "image_url": {
            "type": "string",
            "description": "The URL of the image to analyze. Optional."
        },
        "image_path": {
            "type": "string",
            "description": "The local file path of the image to analyze. Optional."
        },
        "image_base64": {
            "type": "string",
            "description": "The base64-encoded image data to analyze. Optional."
        },
        "pdf_path": {
            "type": "string",
            "description": "The local file path of the PDF document to analyze. Optional."
        }
    }
    required = ["prompt"]

    def __init__(self, api_key, model="openai/gpt-4o-mini"):
        super().__init__()
        self.api_key = api_key
        self.model = model

    def __call__(self, prompt, image_url=None, image_path=None, image_base64=None, pdf_path=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Handle image or PDF input
        if image_url:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        elif image_path:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64}"
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        elif image_base64:
            data_url = f"data:image/jpeg;base64,{image_base64}"
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
        elif pdf_path:
            with open(pdf_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:application/pdf;base64,{b64}"
            messages[0]["content"].append({
                "type": "file",
                "file": {
                    "filename": pdf_path.split("/")[-1],
                    "file_data": data_url
                }
            })

        payload = {
            "model": self.model,
            "messages": messages
        }

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=payload)
        try:
            return response.json()
        except Exception as e:
            return {"error": f"Failed to parse OpenRouter response: {e}", "raw": response.text} 