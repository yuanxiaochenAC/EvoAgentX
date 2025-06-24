import asyncio
from typing import Any, Callable, Optional, Sequence, List, Tuple

from llama_index.core.async_utils import run_jobs
from llama_index.core.schema import TransformComponent, BaseNode, MetadataMode
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)

from evoagentx.models.base_model import BaseLLM
from evoagentx.prompts.rag.graph_extract import ENTITY_EXTRACT_PROMPT, RELATION_EXTRACT_PROMPT


class BasicGraphExtractLLM(TransformComponent):
    """This Transform for those llm without tools ability to build the GraphIndexing."""
    llm: BaseLLM
    extract_prompt: str
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self, 
        llm: BaseLLM,
        entity_extract_prompt: Optional[str] = None,
        relation_extract_prompt: Optional[str] = None,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ):

        entity_extract_prompt = entity_extract_prompt or ENTITY_EXTRACT_PROMPT
        relation_extract_prompt = relation_extract_prompt or RELATION_EXTRACT_PROMPT

        super().__init__(
            llm=llm,
            entity_extract_prompt=entity_extract_prompt,
            relation_extract_prompt=relation_extract_prompt,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode=MetadataMode.LLM)     # Output the format content, the metadata and text are included.
        try:
            # format entity prompt
            extract_prompt = self.entity_extract_prompt.format_map({"text": text})
            llm_responses = self.llm.generate(
                prompt=extract_prompt,
            )
        except ValueError:
            triples = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triples:
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

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