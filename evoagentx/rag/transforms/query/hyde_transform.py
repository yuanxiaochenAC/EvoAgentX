from string import Template
from typing import Dict, Union, Optional

from evoagentx.rag.schema import Query
from evoagentx.models.base_model import BaseLLM
from evoagentx.prompts.rag.hyde import DEFAULT_HYDE_PROMPT

class HyDETransform:
    """
    Hypothetical Document Embeddings (HyDE) query transform.

    It uses an LLM to generate hypothetical answer(s) to a given query,
    and use the resulting documents as embedding strings.

    As described in `[Precise Zero-Shot Dense Retrieval without Relevance Labels]
    (https://arxiv.org/abs/2212.10496)`
    """
    def __init__(
        self,
        llm: BaseLLM,
        hyde_prompt: Optional[Union[str, Template]] = None,
        include_original = True
    ) -> None:
        self._llm = llm

        self._hyde_prompt = hyde_prompt or DEFAULT_HYDE_PROMPT
        self._include_original = include_original
    
    def _run(self, query: Query, metadata: Dict) -> Query:
        query_str = query.query_str

        # Format the query for LLM
        instrutction = DEFAULT_HYDE_PROMPT.safe_substitute(query=query_str)

        # generate Hyde
        hypothetical_doc = self._llm.single_generate(instrutction)
        embedding_strs = [hypothetical_doc]
        if self._include_original:
            embedding_strs.extend(query.embedding_strs)
        
        # Construct a New Query instance
        tmp_query = query.deepcopy()
        tmp_query.custom_embedding_strs = embedding_strs
        return tmp_query

    def run(
        self,
        query_or_str: Union[str, Query],
        metadata: Optional[Dict] = None,
    ) -> Query:
        """Run query transform."""
        metadata = metadata or {}
        if isinstance(query_or_str, str):
            query = Query(
                query_str=query_or_str,
                custom_embedding_strs=[query_or_str],
            )
        else:
            query = query_or_str

        return self._run(query, metadata=metadata)

    def __call__(
        self,
        query_bundle_or_str: Union[str, Query],
        metadata: Optional[Dict] = None,
    ) -> Query:
        """Run query processor."""
        return self.run(query_bundle_or_str, metadata=metadata)