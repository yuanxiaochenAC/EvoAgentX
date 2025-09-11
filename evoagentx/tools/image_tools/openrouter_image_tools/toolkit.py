from typing import Optional
from ...tool import Toolkit
from .image_generation import OpenRouterImageGenerationEditTool
from .image_analysis import ImageAnalysisTool


class OpenRouterImageToolkit(Toolkit):
    def __init__(self, name: str = "OpenRouterImageToolkit", api_key: Optional[str] = None):
        analysis = ImageAnalysisTool(api_key=api_key)
        generation = OpenRouterImageGenerationEditTool(api_key=api_key)
        super().__init__(name=name, tools=[analysis, generation])
        self.api_key = api_key


class ImageAnalysisToolkit(Toolkit):
    def __init__(self, name: str = "ImageAnalysisToolkit", api_key: Optional[str] = None, model: str = "openai/gpt-4o"):
        # Lightweight re-export in this consolidated file
        analysis = ImageAnalysisTool(api_key=api_key, model=model)
        super().__init__(name=name, tools=[analysis])
        self.api_key = api_key
        self.model = model


