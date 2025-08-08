"""
Example demonstrating the improved storage architecture.

This example shows how the storage system now properly handles both local and remote storage
without making assumptions about filesystem operations.
"""

from .storage_file import LocalStorageHandler
from .storage_supabase import SupabaseStorageHandler
from .storage_base import StorageBase


def example_local_storage():
    """Example using local storage with automatic directory creation."""
    print("=== Local Storage Example ===")
    
    # Local storage will create directories automatically
    local_storage = LocalStorageHandler(base_path="./workplace/local")
    
    # Test file operations
    result = local_storage.save("test.txt", "Hello from local storage!")
    print(f"Local save result: {result['success']}")
    
    read_result = local_storage.read("test.txt")
    print(f"Local read result: {read_result['success']}")
    if read_result['success']:
        print(f"Content: {read_result['content']}")


def example_remote_storage():
    """Example using remote storage without local filesystem operations."""
    print("\n=== Remote Storage Example ===")
    
    try:
        # Remote storage won't create local directories
        remote_storage = SupabaseStorageHandler(
            bucket_name="documents",
            base_path="/workplace/remote"
        )
        
        # Test file operations (requires valid Supabase credentials)
        result = remote_storage.save("test.txt", "Hello from remote storage!")
        print(f"Remote save result: {result['success']}")
        
        read_result = remote_storage.read("test.txt")
        print(f"Remote read result: {read_result['success']}")
        if read_result['success']:
            print(f"Content: {read_result['content']}")
            
    except Exception as e:
        print(f"Remote storage example (requires Supabase setup): {str(e)}")


def example_storage_base():
    """Example using the abstract base class."""
    print("\n=== Storage Base Example ===")
    
    # StorageBase doesn't assume any filesystem operations
    base_storage = StorageBase(base_path="./workplace/base")
    
    # This won't create local directories
    print(f"Base storage initialized with path: {base_storage.base_path}")
    print("Base storage doesn't create local directories (as expected)")


def example_mixed_storage():
    """Example showing how different storage types handle initialization."""
    print("\n=== Mixed Storage Example ===")
    
    # Local storage - creates directories
    local_storage = LocalStorageHandler(base_path="./workplace/mixed/local")
    print("✅ Local storage initialized (creates directories)")
    
    # Remote storage - doesn't create local directories
    try:
        remote_storage = SupabaseStorageHandler(
            bucket_name="documents",
            base_path="/workplace/mixed/remote"
        )
        print("✅ Remote storage initialized (no local directories)")
    except Exception as e:
        print(f"⚠️ Remote storage example (requires setup): {str(e)}")
    
    # Base storage - doesn't create directories
    base_storage = StorageBase(base_path="./workplace/mixed/base")
    print("✅ Base storage initialized (no local directories)")


def demonstrate_path_resolution():
    """Demonstrate how different storage types resolve paths."""
    print("\n=== Path Resolution Examples ===")
    
    # Local storage path resolution
    local_storage = LocalStorageHandler(base_path="./workplace")
    local_path = local_storage._resolve_path("test.txt")
    print(f"Local path resolution: 'test.txt' -> '{local_path}'")
    
    # Remote storage path resolution
    remote_storage = SupabaseStorageHandler(base_path="/workplace")
    remote_path = remote_storage._resolve_path("test.txt")
    print(f"Remote path resolution: 'test.txt' -> '{remote_path}'")
    
    # Base storage path resolution
    base_storage = StorageBase(base_path="./workplace")
    base_path = base_storage._resolve_path("test.txt")
    print(f"Base path resolution: 'test.txt' -> '{base_path}'")


if __name__ == "__main__":
    print("Storage Architecture Demonstration")
    print("=" * 50)
    
    example_local_storage()
    example_remote_storage()
    example_storage_base()
    example_mixed_storage()
    demonstrate_path_resolution()
    
    print("\n" + "=" * 50)
    print("Architecture Benefits:")
    print("- Local storage creates directories automatically")
    print("- Remote storage doesn't attempt local filesystem operations")
    print("- Base class is storage-agnostic")
    print("- Each storage type handles initialization appropriately")
    print("- No errors when mixing local and remote storage")
