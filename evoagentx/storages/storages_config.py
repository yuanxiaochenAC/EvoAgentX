from typing import Optional

from pydantic import Field

from ..core.base_config import BaseConfig


class DBConfig(BaseConfig):
    """
    """
    db_name: str = Field(default="sqlite", description="")
    path: Optional[str] = Field(default="", description="")
    ip: Optional[str] = Field(default="", description="")
    port: Optional[str] = Field(default="", description="")


class VectorStoreConfig(BaseConfig):
    """
    """
