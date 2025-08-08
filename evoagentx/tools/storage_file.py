from .tool import Tool, Toolkit
from .storage_base import StorageBase
from typing import Dict, Any, List, Optional
from ..core.logging import logger
import os
import shutil
from pathlib import Path
from datetime import datetime


class LocalStorageHandler(StorageBase):
    """
    Local filesystem storage implementation.
    Provides all file operations for local storage with default working directory.
    """
    
    def __init__(self, base_path: str = ".", **kwargs):
        """
        Initialize local storage handler.
        
        Args:
            base_path (str): Base directory for storage operations (default: current directory)
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(base_path=base_path, **kwargs)
    
    def _initialize_storage(self):
        """Initialize local storage - create base directory if it doesn't exist"""
        # Convert base_path to Path for local filesystem operations
        base_path = Path(self.base_path)
        # Ensure base directory exists
        base_path.mkdir(parents=True, exist_ok=True)
    
    def _resolve_path(self, file_path: str) -> str:
        """Resolve file path for local filesystem"""
        path = Path(file_path)
        if not path.is_absolute():
            # If it's a relative path, prepend the base path
            path = Path(self.base_path) / path
        return str(path)
    
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content from local filesystem"""
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            raise
    
    def _write_raw(self, path: str, content: bytes, **kwargs) -> bool:
        """Write raw file content to local filesystem"""
        try:
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing file {path}: {str(e)}")
            return False
    
    def _delete_raw(self, path: str) -> bool:
        """Delete file or directory from local filesystem"""
        try:
            path_obj = Path(path)
            if path_obj.is_file():
                path_obj.unlink()
            elif path_obj.is_dir():
                shutil.rmtree(path_obj)
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Error deleting {path}: {str(e)}")
            return False
    
    def _list_raw(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List files and directories in local filesystem"""
        try:
            if path is None:
                path = str(self.base_path)
            
            path_obj = Path(path)
            if not path_obj.exists() or not path_obj.is_dir():
                return []
            
            items = []
            
            def scan_directory(current_path: Path, current_depth: int):
                if current_depth > max_depth:
                    return
                
                try:
                    for item in current_path.iterdir():
                        # Skip hidden files if not included
                        if not include_hidden and item.name.startswith('.'):
                            continue
                        
                        try:
                            stat = item.stat()
                            item_info = {
                                "name": item.name,
                                "path": str(item),
                                "type": "directory" if item.is_dir() else "file",
                                "size_bytes": stat.st_size if item.is_file() else 0,
                                "size_mb": round(stat.st_size / (1024 * 1024), 2) if item.is_file() else 0,
                                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "extension": item.suffix.lower() if item.is_file() else "",
                                "is_hidden": item.name.startswith('.')
                            }
                            
                            items.append(item_info)
                            
                            # Recursively scan subdirectories
                            if item.is_dir() and current_depth < max_depth:
                                scan_directory(item, current_depth + 1)
                                
                        except (PermissionError, OSError):
                            # Skip files we can't access
                            continue
                            
                except (PermissionError, OSError) as e:
                    logger.warning(f"Error scanning directory {current_path}: {str(e)}")
            
            scan_directory(path_obj, 0)
            return items
            
        except Exception as e:
            logger.error(f"Error listing directory {path}: {str(e)}")
            return []
    
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists in local filesystem"""
        return Path(path).exists()
    
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory in local filesystem"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {path}: {str(e)}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive information about a file"""
        try:
            resolved_path = self._resolve_path(file_path)
            path_obj = Path(resolved_path)
            
            if not path_obj.exists():
                return {"success": False, "error": f"File {file_path} does not exist"}
            
            stat = path_obj.stat()
            return {
                "success": True,
                "file_path": resolved_path,
                "file_name": path_obj.name,
                "file_extension": path_obj.suffix.lower(),
                "mime_type": self.get_mime_type(resolved_path),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_file": path_obj.is_file(),
                "is_directory": path_obj.is_dir(),
                "is_readable": os.access(path_obj, os.R_OK),
                "is_writable": os.access(path_obj, os.W_OK),
                "exists": True
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class SaveTool(Tool):
    name: str = "save"
    description: str = "Save content to a file with automatic format detection and support for various file types including documents, data files, images, videos, and sound files"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to save"
        },
        "content": {
            "type": "string",
            "description": "Content to save to the file (can be JSON string for structured data)"
        },
        "encoding": {
            "type": "string",
            "description": "Text encoding for text files (default: utf-8)"
        },
        "indent": {
            "type": "integer",
            "description": "Indentation for JSON files (default: 2)"
        },
        "sheet_name": {
            "type": "string",
            "description": "Sheet name for Excel files (default: Sheet1)"
        },
        "root_tag": {
            "type": "string",
            "description": "Root tag for XML files (default: root)"
        }
    }
    required: Optional[List[str]] = ["file_path", "content"]

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, file_path: str, content: str, encoding: str = "utf-8", indent: int = 2, 
                 sheet_name: str = "Sheet1", root_tag: str = "root") -> Dict[str, Any]:
        """
        Save content to a file with automatic format detection.
        
        Args:
            file_path: Path to the file to save
            content: Content to save to the file
            encoding: Text encoding for text files
            indent: Indentation for JSON files
            sheet_name: Sheet name for Excel files
            root_tag: Root tag for XML files
            
        Returns:
            Dictionary containing the save operation result
        """
        try:
            # Parse content based on file type
            file_extension = self.storage_handler.get_file_type(file_path)
            parsed_content = content
            
            # Try to parse JSON content for appropriate file types
            if file_extension in ['.json', '.yaml', '.yml', '.xml']:
                try:
                    import json
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    parsed_content = content
            
            # Handle CSV content
            elif file_extension == '.csv':
                try:
                    import json
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "CSV content must be a list of dictionaries"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "CSV content must be valid JSON array"}
            
            # Handle Excel content
            elif file_extension == '.xlsx':
                try:
                    import json
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "Excel content must be a list of lists"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "Excel content must be valid JSON array"}
            
            kwargs = {
                "encoding": encoding,
                "indent": indent,
                "sheet_name": sheet_name,
                "root_tag": root_tag
            }
            
            result = self.storage_handler.save(file_path, parsed_content, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in SaveTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class ReadTool(Tool):
    name: str = "read"
    description: str = "Read content from a file with automatic format detection and support for various file types including documents, data files, images, videos, and sound files"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read"
        },
        "encoding": {
            "type": "string",
            "description": "Text encoding for text files (default: utf-8)"
        },
        "sheet_name": {
            "type": "string",
            "description": "Sheet name for Excel files (optional)"
        },
        "head": {
            "type": "integer",
            "description": "Number of characters to return from the beginning of the file (default: 0 means return everything)"
        }
    }
    required: Optional[List[str]] = ["file_path"]

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, file_path: str, encoding: str = "utf-8", sheet_name: str = None, head: int = 0) -> Dict[str, Any]:
        """
        Read content from a file with automatic format detection.
        
        Args:
            file_path: Path to the file to read
            encoding: Text encoding for text files
            sheet_name: Sheet name for Excel files
            head: Number of characters to return from the beginning
            
        Returns:
            Dictionary containing the read operation result
        """
        try:
            kwargs = {
                "encoding": encoding,
                "sheet_name": sheet_name,
                "head": head
            }
            
            result = self.storage_handler.read(file_path, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ReadTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class AppendTool(Tool):
    name: str = "append"
    description: str = "Append content to a file (only for supported formats: txt, json, csv, yaml, pickle, xlsx)"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to append to"
        },
        "content": {
            "type": "string",
            "description": "Content to append to the file (can be JSON string for structured data)"
        },
        "encoding": {
            "type": "string",
            "description": "Text encoding for text files (default: utf-8)"
        },
        "sheet_name": {
            "type": "string",
            "description": "Sheet name for Excel files (optional)"
        }
    }
    required: Optional[List[str]] = ["file_path", "content"]

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, file_path: str, content: str, encoding: str = "utf-8", sheet_name: str = None) -> Dict[str, Any]:
        """
        Append content to a file with automatic format detection.
        
        Args:
            file_path: Path to the file to append to
            content: Content to append to the file
            encoding: Text encoding for text files
            sheet_name: Sheet name for Excel files
            
        Returns:
            Dictionary containing the append operation result
        """
        try:
            # Parse content based on file type
            file_extension = self.storage_handler.get_file_type(file_path)
            parsed_content = content
            
            # Try to parse JSON content for appropriate file types
            if file_extension in ['.json', '.yaml', '.yml']:
                try:
                    import json
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    parsed_content = content
            
            # Handle CSV content
            elif file_extension == '.csv':
                try:
                    import json
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "CSV content must be a list of dictionaries"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "CSV content must be valid JSON array"}
            
            # Handle Excel content
            elif file_extension == '.xlsx':
                try:
                    import json
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "Excel content must be a list of lists"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "Excel content must be valid JSON array"}
            
            kwargs = {
                "encoding": encoding,
                "sheet_name": sheet_name
            }
            
            result = self.storage_handler.append(file_path, parsed_content, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AppendTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class DeleteTool(Tool):
    name: str = "delete"
    description: str = "Delete a file or directory"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to the file or directory to delete"
        }
    }
    required: Optional[List[str]] = ["path"]

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, path: str) -> Dict[str, Any]:
        """
        Delete a file or directory.
        
        Args:
            path: Path to the file or directory to delete
            
        Returns:
            Dictionary containing the delete operation result
        """
        try:
            result = self.storage_handler.delete(path)
            return result
            
        except Exception as e:
            logger.error(f"Error in DeleteTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class MoveTool(Tool):
    name: str = "move"
    description: str = "Move or rename a file or directory"
    inputs: Dict[str, Dict[str, str]] = {
        "source": {
            "type": "string",
            "description": "Source path of the file or directory to move"
        },
        "destination": {
            "type": "string",
            "description": "Destination path where to move the file or directory"
        }
    }
    required: Optional[List[str]] = ["source", "destination"]

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Move or rename a file or directory.
        
        Args:
            source: Source path of the file or directory to move
            destination: Destination path where to move the file or directory
            
        Returns:
            Dictionary containing the move operation result
        """
        try:
            result = self.storage_handler.move(source, destination)
            return result
            
        except Exception as e:
            logger.error(f"Error in MoveTool: {str(e)}")
            return {"success": False, "error": str(e), "source": source, "destination": destination}


class CopyTool(Tool):
    name: str = "copy"
    description: str = "Copy a file"
    inputs: Dict[str, Dict[str, str]] = {
        "source": {
            "type": "string",
            "description": "Source path of the file to copy"
        },
        "destination": {
            "type": "string",
            "description": "Destination path where to copy the file"
        }
    }
    required: Optional[List[str]] = ["source", "destination"]

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a file.
        
        Args:
            source: Source path of the file to copy
            destination: Destination path where to copy the file
            
        Returns:
            Dictionary containing the copy operation result
        """
        try:
            result = self.storage_handler.copy(source, destination)
            return result
            
        except Exception as e:
            logger.error(f"Error in CopyTool: {str(e)}")
            return {"success": False, "error": str(e), "source": source, "destination": destination}


class CreateDirectoryTool(Tool):
    name: str = "create_directory"
    description: str = "Create a directory"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path of the directory to create"
        }
    }
    required: Optional[List[str]] = ["path"]

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, path: str) -> Dict[str, Any]:
        """
        Create a directory.
        
        Args:
            path: Path of the directory to create
            
        Returns:
            Dictionary containing the create directory operation result
        """
        try:
            result = self.storage_handler.create_directory(path)
            return result
            
        except Exception as e:
            logger.error(f"Error in CreateDirectoryTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class ListFileTool(Tool):
    name: str = "list_files"
    description: str = "List files and directories in a path with structured information"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to list files from (default: current working directory)"
        },
        "max_depth": {
            "type": "integer",
            "description": "Maximum depth to traverse (default: 3)"
        },
        "include_hidden": {
            "type": "boolean",
            "description": "Include hidden files and directories (default: false)"
        }
    }
    required: Optional[List[str]] = []

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        """
        List files and directories in a path.
        
        Args:
            path: Path to list files from
            max_depth: Maximum depth to traverse
            include_hidden: Include hidden files and directories
            
        Returns:
            Dictionary containing the list operation result
        """
        try:
            result = self.storage_handler.list(path, max_depth=max_depth, include_hidden=include_hidden)
            return result
            
        except Exception as e:
            logger.error(f"Error in ListFileTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class ExistsTool(Tool):
    name: str = "exists"
    description: str = "Check if a file or directory exists"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to check for existence"
        }
    }
    required: Optional[List[str]] = ["path"]

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, path: str) -> Dict[str, Any]:
        """
        Check if a file or directory exists.
        
        Args:
            path: Path to check for existence
            
        Returns:
            Dictionary containing the existence check result
        """
        try:
            exists = self.storage_handler.exists(path)
            return {
                "success": True,
                "path": path,
                "exists": exists
            }
            
        except Exception as e:
            logger.error(f"Error in ExistsTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class ListSupportedFormatsTool(Tool):
    name: str = "list_supported_formats"
    description: str = "List all supported file formats and their capabilities"
    inputs: Dict[str, Dict[str, str]] = {}
    required: Optional[List[str]] = []

    def __init__(self, storage_handler: StorageBase = None):
        super().__init__()
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self) -> Dict[str, Any]:
        """
        List all supported file formats and their capabilities.
        
        Returns:
            Dictionary containing supported formats information
        """
        try:
            result = self.storage_handler.get_supported_formats()
            return result
            
        except Exception as e:
            logger.error(f"Error in ListSupportedFormatsTool: {str(e)}")
            return {"success": False, "error": str(e)}


class StorageToolkit(Toolkit):
    """
    Comprehensive storage toolkit with local filesystem operations.
    Provides tools for reading, writing, appending, deleting, moving, copying files,
    creating directories, and listing files with support for various file formats.
    """
    
    def __init__(self, name: str = "StorageToolkit", base_path: str = "./workplace/storage", storage_handler: StorageBase = None):
        """
        Initialize the storage toolkit.
        
        Args:
            name: Name of the toolkit
            base_path: Base directory for storage operations (default: ./workplace/storage)
            storage_handler: Storage handler instance (defaults to LocalStorageHandler)
        """
        # Create the shared storage handler instance
        if storage_handler is None:
            storage_handler = LocalStorageHandler(base_path=base_path)
        
        # Create tools with the storage handler
        tools = [
            SaveTool(storage_handler=storage_handler),
            ReadTool(storage_handler=storage_handler),
            AppendTool(storage_handler=storage_handler),
            DeleteTool(storage_handler=storage_handler),
            MoveTool(storage_handler=storage_handler),
            CopyTool(storage_handler=storage_handler),
            CreateDirectoryTool(storage_handler=storage_handler),
            ListFileTool(storage_handler=storage_handler),
            ExistsTool(storage_handler=storage_handler),
            ListSupportedFormatsTool(storage_handler=storage_handler)
        ]
        
        super().__init__(name=name, tools=tools)
        self.storage_handler = storage_handler 