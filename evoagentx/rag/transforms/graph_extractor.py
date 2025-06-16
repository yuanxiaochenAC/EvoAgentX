import asyncio
from typing import Any, Callable, Optional, Sequence, Union

from llama_index.core.schema import BaseNode
from llama_index.core.async_utils import run_jobs
from llama_index.core.schema import TransformComponent

from evoagentx.models.base_model import BaseLLM

class BasicGraphExtractLLM(TransformComponent):
    """This Transform for those llm without tools ability to build the GraphIndexing."""
    def __init__(
        self, 
        llm: BaseLLM,
        num_workers: int = 4,
        max_paths_per_chunk: int = 5, 
    ):
        ...
    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))
    
    async def acall(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

    @classmethod
    def class_name(cls) -> str:
        return "BasicGraphExtractLLM"

class GraphExtractLLM(TransformComponent):
    def __init__(
        self, 
        llm: BaseLLM,
        num_workers: int = 4,
        max_paths_per_chunk: int = 5, 
    ):
        ...
        self.llm = llm
        self.num_workers = num_workers
        self.max_paths_per_chunk = max_paths_per_chunk

    def _extract_triples(
        self, node: BaseNode, show_progress: bool = False, **kwargs: Any
    ) -> BaseNode:
        """Extract triples from node async."""
        assert hasattr(node, "text")


    def _extract_relation(
        self, node: BaseNode, show_progress: bool = False, **kwargs: Any
    ) -> BaseNode:
        """Extract triples from node async."""
        ...

    async def _aextract(
        self, node: BaseNode, show_progress: bool = False, **kwargs: Any
    ) -> BaseNode:
        """Extract triples and relation from node async."""
        assert hasattr(node, "text")

        self._extract_triples()
        
        self._extract_relation()

    async def acall(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractLLM"
    

if __name__ == "__main__":
    from evoagentx.models import OpenAILLMConfig, OpenAILLM

    import os
    import dotenv
    dotenv.load_dotenv()

    # Configure the model
    config = OpenAILLMConfig(
        model="gpt-4o-mini",  
        openai_key=os.environ["OPENAI_API_KEY"],
        temperature=0.7,
        max_tokens=1000
    )

    # Initialize the model
    llm = OpenAILLM(config=config)

    # Generate text
    response = llm.single_generate(
        [{"role": "system", "content": ""}, {"role": "user", "content": "", "tools": []}]
    )
    import pdb;pdb.set_trace()