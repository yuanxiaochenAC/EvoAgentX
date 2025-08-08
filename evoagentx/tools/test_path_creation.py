"""
Test script to verify path creation functionality in storage handlers.
"""

import os
import tempfile
import shutil
from pathlib import Path

from .storage_file import LocalStorageHandler
from .storage_base import StorageBase


def test_local_storage_path_creation():
    """Test that LocalStorageHandler creates paths automatically."""
    print("Testing LocalStorageHandler path creation...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test path that doesn't exist
        test_path = os.path.join(temp_dir, "workplace", "storage", "test")
        
        # Create storage handler with non-existent path
        storage = LocalStorageHandler(base_path=test_path)
        
        # Check if path was created
        if os.path.exists(test_path):
            print(f"✅ Path created successfully: {test_path}")
        else:
            print(f"❌ Path was not created: {test_path}")
            return False
        
        # Test file operations
        test_file = os.path.join(test_path, "test.txt")
        result = storage.save(test_file, "Hello, World!")
        
        if result["success"]:
            print(f"✅ File saved successfully: {test_file}")
        else:
            print(f"❌ File save failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Test reading the file
        read_result = storage.read(test_file)
        if read_result["success"] and read_result["content"] == "Hello, World!":
            print(f"✅ File read successfully: {test_file}")
        else:
            print(f"❌ File read failed: {read_result.get('error', 'Unknown error')}")
            return False
    
    return True


def test_storage_base_initialization():
    """Test that StorageBase initializes without filesystem operations."""
    print("\nTesting StorageBase initialization...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test path that doesn't exist
        test_path = os.path.join(temp_dir, "workplace", "base", "test")
        
        # Create storage handler with non-existent path
        # StorageBase should not create local directories
        storage = StorageBase(base_path=test_path)
        
        # Check that base_path is stored as string
        if isinstance(storage.base_path, str):
            print(f"✅ Base path stored as string: {storage.base_path}")
        else:
            print(f"❌ Base path not stored as string: {type(storage.base_path)}")
            return False
        
        # Check that path was NOT created (StorageBase doesn't do local filesystem ops)
        if not os.path.exists(test_path):
            print(f"✅ Path not created (as expected for StorageBase): {test_path}")
        else:
            print(f"❌ Path was created unexpectedly: {test_path}")
            return False
    
    return True


def test_workplace_directory_structure():
    """Test that workplace directories are created with proper structure."""
    print("\nTesting workplace directory structure...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        workplace_path = os.path.join(temp_dir, "workplace")
        
        # Test different workplace directories
        workplace_dirs = [
            "storage",
            "python", 
            "docker",
            "images",
            "analysis",
            "cmd"
        ]
        
        all_created = True
        
        for dir_name in workplace_dirs:
            dir_path = os.path.join(workplace_path, dir_name)
            
            # Create storage handler
            storage = LocalStorageHandler(base_path=dir_path)
            
            # Check if directory was created
            if os.path.exists(dir_path):
                print(f"✅ Created: {dir_path}")
            else:
                print(f"❌ Failed to create: {dir_path}")
                all_created = False
        
        if all_created:
            print("✅ All workplace directories created successfully")
        else:
            print("❌ Some workplace directories failed to create")
            return False
    
    return True


def main():
    """Run all path creation tests."""
    print("=" * 50)
    print("Testing Storage Handler Path Creation")
    print("=" * 50)
    
    tests = [
        test_local_storage_path_creation,
        test_storage_base_initialization,
        test_workplace_directory_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error: {test.__name__} - {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! Path creation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    main()
