from evoagentx.tools.tool import Tool
from evoagentx.rag.rag import RAGEngine


class RagTool(Tool):
    def __init__(self, name: str = "RagTool", search_engine: RAGEngine = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.search_engine = search_engine