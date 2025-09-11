from ...tool import Toolkit
from .image_generation import OpenAIImageGenerationTool
from .image_edit import OpenAIImageEditTool
from .image_analysis_openai import OpenAIImageAnalysisTool


class OpenAIImageToolkit(Toolkit):
    def __init__(self, name: str = "OpenAIImageToolkit", api_key: str = None, organization_id: str = None,
                 generation_model: str = "dall-e-3", save_path: str = "./generated_images"):
        gen_tool = OpenAIImageGenerationTool(api_key=api_key, organization_id=organization_id,
                                             model=generation_model, save_path=save_path)
        edit_tool = OpenAIImageEditTool(api_key=api_key, organization_id=organization_id,
                                        save_path=save_path)
        analysis_tool = OpenAIImageAnalysisTool(api_key=api_key, organization_id=organization_id)
        super().__init__(name=name, tools=[gen_tool, edit_tool, analysis_tool])
        self.api_key = api_key
        self.organization_id = organization_id
        self.generation_model = generation_model
        self.save_path = save_path


