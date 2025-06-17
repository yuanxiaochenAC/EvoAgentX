from evoagentx.tools.tool import Tool
from evoagentx.rag.search_engine import SearchEngine


class RagTool(Tool):
    def __init__(self, name: str = "RagTool", search_engine: SearchEngine = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.search_engine = search_engine