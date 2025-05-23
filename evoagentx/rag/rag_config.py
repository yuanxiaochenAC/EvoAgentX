from pydantic import BaseModel, Field
from typing import Optional, Union, List

from ..core.base_config import BaseConfig


class EmbeddingConfig(BaseConfig):
    pass


class TokenizerConfig(BaseConfig):
    pass


class RAGConfig(BaseConfig):
    pass

