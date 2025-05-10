from typing import Optional
from dataclasses import dataclass

from pydantic import Field

from ..core.base_config import BaseConfig


@dataclass
class DBConfig(BaseConfig):
    """
    """
    db_name: str
    path: Optional[str] = Field(default=None, description="The path for file sotrage system.")
    ip: Optional[str] = Field(default=None, description="The ip")
    port: Optional[str] = Field(default=None, description="")


@dataclass
class VectorStoreConfig(BaseConfig):
    """
    """
    