from string import Template
from typing import Dict, Union, Optional

from evoagentx.rag.schema import Query
from evoagentx.models.base_model import BaseLLM
from evoagentx.rag.transforms.query.base import BaseQueryTransform
from evoagentx.prompts.rag.hyde import DEFAULT_HYDE_PROMPT, HYDE_SYSTEM_IMPLE_


class HyDETransform(BaseQueryTransform):
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
        instruction = self._hyde_prompt.format_map({"query": query_str})

        # generate Hyde
        hypothetical_doc = self._llm.generate(
            prompt=instruction,
            system_message=HYDE_SYSTEM_IMPLE_
        ).content

        embedding_strs = [hypothetical_doc]
        if self._include_original:
            embedding_strs.extend(query.embedding_strs)

        # Construct a New Query instance
        tmp_query = query.deepcopy()
        tmp_query.custom_embedding_strs = embedding_strs
        return tmp_query


if __name__ == "__main__":
    # Configure the model
    import dotenv
    dotenv.load_dotenv()
    from evoagentx.models import OpenAILLMConfig, OpenAILLM
    
    import os
    os.environ["SSL_CERT_FILE"] = r"D:\miniconda3\envs\envoagentx\Library\ssl\cacert.pem"
    config = OpenAILLMConfig(
        model="gpt-4o-mini",  
        temperature=0.7,
        max_tokens=1000,
        openai_key=os.environ["OPENAI_API_KEY"]
    )

    # Initialize the model
    llm = OpenAILLM(config=config)
    HyDETrans = HyDETransform(llm)
    output_query = HyDETrans(Query(query_str="Were Scott Derrickson and Ed Wood of the same nationality?"))
    print(output_query)