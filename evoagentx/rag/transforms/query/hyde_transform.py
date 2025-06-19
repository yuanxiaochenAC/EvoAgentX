from string import Template
from typing import Dict, Union, Optional

from .base import BaseTransform
from evoagentx.rag.schema import Query
from evoagentx.models.base_model import BaseLLM
from evoagentx.prompts.rag.hyde import DEFAULT_HYDE_PROMPT


class HyDETransform(BaseTransform):
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