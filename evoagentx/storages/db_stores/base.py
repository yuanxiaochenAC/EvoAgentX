from abc import ABC, abstractmethod


class DBStoreBase(ABC):
    @abstractmethod
    def insert(self, metadata, database=None, table=None):
        """Insert metadata into a collection."""
        pass

    @abstractmethod
    def delete(self, metadata_id, database=None, table=None):
        """Delete a metadata by ID."""
        pass

    @abstractmethod
    def update(self, metadata_id, new_metadata=None, database=None, table=None):
        """Update a new_metadata by ID."""
        pass

    @abstractmethod
    def get(self, metadata_id, database=None, table=None):
        """Retrieve a metadata by ID."""
        pass

    @abstractmethod
    def col_info(self):
        """Get information about a collection."""
        pass