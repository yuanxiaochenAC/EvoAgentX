from typing import Dict, Optional, List
from ...tool import Tool
from .openai_utils import create_openai_client


class OpenAIImageAnalysisTool(Tool):
    name: str = "openai_image_analysis"
    description: str = "Simple image analysis via OpenAI Responses API (input_text + input_image)."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "User question/instruction. Required."},
        "image_url": {"type": "string", "description": "HTTP(S) image URL. Optional if image_path provided."},
        "image_path": {"type": "string", "description": "Local image path; converted to data URL internally."},
        "model": {"type": "string", "description": "OpenAI model for responses.create (e.g., gpt-4o-mini, gpt-4.1, gpt-5). Optional."},
    }
    required: Optional[List[str]] = ["prompt"]

    def __init__(self, api_key: str, organization_id: str = None, model: str = "gpt-4o-mini"):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model

    def __call__(
        self,
        prompt: str,
        image_url: str = None,
        image_path: str = None,
        model: str = None,
    ):
        try:
            client = create_openai_client(self.api_key, self.organization_id)
            actual_model = model if model else self.model

            # Resolve image source: prefer URL, else local path to data URL
            final_image_url = image_url
            if not final_image_url and image_path:
                import base64, mimetypes
                mime, _ = mimetypes.guess_type(image_path)
                mime = mime or "image/png"
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                final_image_url = f"data:{mime};base64,{b64}"

            response = client.responses.create(
                model=actual_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": final_image_url},
                        ],
                    }
                ],
            )

            # Prefer unified output_text when present
            text = getattr(response, "output_text", None)
            if text is None:
                # Fallback: try to assemble from content if SDK shape differs
                try:
                    choices = getattr(response, "output", None) or getattr(response, "choices", None)
                    if choices and isinstance(choices, list):
                        first = choices[0]
                        text = getattr(first, "message", {}).get("content", "") if isinstance(first, dict) else ""
                except Exception:
                    text = ""

            return {"content": text or ""}
        except Exception as e:
            return {"error": f"OpenAI image analysis failed: {e}"}


