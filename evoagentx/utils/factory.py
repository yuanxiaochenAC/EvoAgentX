import importlib

from ..storages.storages_config import DBConfig, VectorStoreConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class DBFactory:
    provider_to_class = {
        "sqlite": "evoagentx.storages.db_stores.sqlite.SQLite",
        "posgre_sql": "evoagentx.storages.db_stores.posgre_sql.",
    }

    @classmethod
    def create(cls, provider_name: str, config: DBConfig):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            db_store_instance = load_class(class_type)
            return db_store_instance(**config)
        else:
            raise ValueError(f"Unsupported Database provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "mem0.vector_stores.qdrant.Qdrant",
        "chroma": "mem0.vector_stores.chroma.ChromaDB",
        "faiss": "mem0.vector_stores.faiss.FAISS",
    }

    @classmethod
    def create(cls, config: VectorStoreConfig):
        ...