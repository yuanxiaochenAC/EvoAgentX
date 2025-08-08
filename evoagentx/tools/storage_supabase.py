import os
import json
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .storage_base import StorageBase
from .tool import Tool, Toolkit
from ..core.logging import logger


class SupabaseStorageHandler(StorageBase):
    """
    Supabase remote storage implementation.
    Provides file operations via Supabase Storage API with environment-based configuration.
    """
    
    def __init__(self, bucket_name: str = None, base_path: str = "/", **kwargs):
        """
        Initialize Supabase storage handler.
        
        Args:
            bucket_name: Supabase storage bucket name (default: from environment or "default")
            base_path: Base path for storage operations (default: "/")
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(base_path=base_path, **kwargs)
        
        # Get bucket name from environment or use default
        self.bucket_name = bucket_name or os.getenv("SUPABASE_BUCKET") or "default"
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase configuration not found in environment variables. "
                "Please set SUPABASE_URL/SUPABASE_KEY environment variables."
            )
        
        # Initialize Supabase client
        try:
            from supabase import create_client, Client
            logger.info(f"Creating Supabase client with URL: {self.supabase_url[:30]}...")
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"Successfully initialized Supabase client for bucket: {bucket_name}")
        except ImportError:
            raise ImportError(
                "Supabase Python client not installed. "
                "Please install it with: pip install supabase"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise Exception(f"Failed to initialize Supabase client: {str(e)}")
    
    def _initialize_storage(self):
        """Initialize remote storage - verify bucket exists and is accessible"""
        try:
            # Test bucket access by listing files (empty list is fine)
            logger.info(f"Testing bucket access for: {self.bucket_name}")
            self.supabase.storage.from_(self.bucket_name).list()
            logger.info(f"Successfully connected to Supabase bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Could not verify bucket access: {str(e)}")
            # Don't raise error as bucket might be empty or have different permissions
    
    def _resolve_path(self, file_path: str) -> str:
        """Resolve file path for remote storage"""
        # For Supabase, we use the base_path as a prefix
        # Remove any leading slash from file_path
        clean_file_path = file_path.lstrip('/')
        
        # If base_path is just "/", don't add it
        if self.base_path == "/":
            return clean_file_path
        
        # Otherwise, combine base_path and file_path
        if not clean_file_path.startswith(self.base_path):
            return f"{self.base_path}/{clean_file_path}".replace('//', '/')
        else:
            return clean_file_path
    
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content from Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Download file from Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).download(file_path)
            
            if isinstance(response, bytes):
                return response
            else:
                # If response is not bytes, try to convert
                return bytes(response) if response else b""
                
        except Exception as e:
            logger.error(f"Error reading file {path} from Supabase: {str(e)}")
            raise
    
    def _write_raw(self, path: str, content: bytes, **kwargs) -> bool:
        """Write raw file content to Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Upload file to Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=content,
                file_options={"content-type": kwargs.get("content_type", "application/octet-stream")}
            )
            
            # Check if upload was successful
            if response and (not isinstance(response, dict) or response.get("error") is None):
                logger.info(f"Successfully uploaded file to Supabase: {file_path}")
                return True
            else:
                logger.error(f"Upload failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error writing file {path} to Supabase: {str(e)}")
            return False
    
    def _delete_raw(self, path: str) -> bool:
        """Delete file from Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Delete file from Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).remove([file_path])
            
            # Check if deletion was successful
            if response and (not isinstance(response, dict) or response.get("error") is None):
                logger.info(f"Successfully deleted file from Supabase: {file_path}")
                return True
            else:
                logger.error(f"Deletion failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting {path} from Supabase: {str(e)}")
            return False
    
    def _list_raw(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List files in Supabase Storage"""
        try:
            # Remove leading slash if present
            list_path = (path or self.base_path).lstrip('/')
            
            # List files from Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).list(list_path)
            
            items = []
            if response and isinstance(response, list):
                for item in response:
                    # Skip hidden files if not included
                    if not include_hidden and item.get('name', '').startswith('.'):
                        continue
                    
                    # Calculate full path
                    full_path = f"{list_path}/{item['name']}" if list_path else item['name']
                    
                    items.append({
                        "name": item.get('name', ''),
                        "path": full_path,
                        "type": "directory" if item.get('metadata', {}).get('mimetype') == 'application/x-directory' else "file",
                        "size_bytes": item.get('metadata', {}).get('size', 0),
                        "size_mb": round(item.get('metadata', {}).get('size', 0) / (1024 * 1024), 2),
                        "modified_time": item.get('updated_at', ''),
                        "extension": Path(item.get('name', '')).suffix.lower(),
                        "is_hidden": item.get('name', '').startswith('.'),
                        "mime_type": item.get('metadata', {}).get('mimetype', '')
                    })
            
            return items
            
        except Exception as e:
            logger.error(f"Error listing directory {path} from Supabase: {str(e)}")
            return []
    
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists in Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Try to get file info
            response = self.supabase.storage.from_(self.bucket_name).list(file_path)
            
            # If we get a response and it's not empty, the file exists
            return bool(response and len(response) > 0)
            
        except Exception:
            return False
    
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory in Supabase Storage"""
        try:
            # Remove leading slash if present
            dir_path = path.lstrip('/')
            
            # Create a placeholder file to establish the directory
            placeholder_content = b"# Directory placeholder"
            placeholder_path = f"{dir_path}/.placeholder"
            
            response = self.supabase.storage.from_(self.bucket_name).upload(
                path=placeholder_path,
                file=placeholder_content,
                file_options={"content-type": "text/plain"}
            )
            
            # Check if upload was successful
            if response and not isinstance(response, dict) or response.get("error") is None:
                return True
            else:
                logger.error(f"Directory creation failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating directory {path} in Supabase: {str(e)}")
            return False
    
    def _save_text(self, file_path: str, content: Any, encoding: str = 'utf-8', **kwargs) -> Dict[str, Any]:
        """Save text content to Supabase Storage"""
        try:
            # Convert content to bytes with specified encoding
            if isinstance(content, str):
                content_bytes = content.encode(encoding)
            else:
                content_bytes = str(content).encode(encoding)
            
            # Use the raw write method
            success = self._write_raw(file_path, content_bytes, **kwargs)
            
            if success:
                return {
                    "success": True,
                    "message": f"Text file saved to Supabase: {file_path}",
                    "file_path": file_path,
                    "content_length": len(content_bytes)
                }
            else:
                return {"success": False, "error": "Failed to upload to Supabase", "file_path": file_path}
                
        except Exception as e:
            logger.error(f"Error saving text file {file_path} to Supabase: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_text(self, file_path: str, encoding: str = 'utf-8', **kwargs) -> Dict[str, Any]:
        """Read text content from Supabase Storage"""
        try:
            # Use the raw read method
            content_bytes = self._read_raw(file_path, **kwargs)
            content = content_bytes.decode(encoding)
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path,
                "content_length": len(content)
            }
        except Exception as e:
            logger.error(f"Error reading text file {file_path} from Supabase: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _save_json(self, file_path: str, content: Any, indent: int = 2, **kwargs) -> Dict[str, Any]:
        """Save JSON content to Supabase Storage"""
        try:
            # Convert content to JSON string
            if isinstance(content, str):
                # If it's already a string, try to parse it to validate JSON
                json.loads(content)
                json_content = content
            else:
                json_content = json.dumps(content, indent=indent, ensure_ascii=False)
            
            # Convert to bytes
            content_bytes = json_content.encode('utf-8')
            
            # Use the raw write method
            success = self._write_raw(file_path, content_bytes, **kwargs)
            
            if success:
                return {
                    "success": True,
                    "message": f"JSON file saved to Supabase: {file_path}",
                    "file_path": file_path
                }
            else:
                return {"success": False, "error": "Failed to upload to Supabase", "file_path": file_path}
                
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path} to Supabase: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def _read_json(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Read JSON content from Supabase Storage"""
        try:
            # Use the raw read method
            content_bytes = self._read_raw(file_path, **kwargs)
            content_str = content_bytes.decode('utf-8')
            
            # Parse JSON
            content = json.loads(content_str)
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path} from Supabase: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive information about a file in Supabase Storage"""
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not self._exists_raw(resolved_path):
                return {"success": False, "error": f"File {file_path} does not exist"}
            
            # Get file content to determine size
            content = self._read_raw(resolved_path)
            
            # Try to get additional metadata from list
            list_path = os.path.dirname(resolved_path.lstrip('/'))
            file_name = os.path.basename(resolved_path)
            
            metadata = {}
            try:
                files = self.supabase.storage.from_(self.bucket_name).list(list_path)
                for file_info in files:
                    if file_info.get('name') == file_name:
                        metadata = file_info.get('metadata', {})
                        break
            except Exception:
                pass
            
            return {
                "success": True,
                "file_path": resolved_path,
                "file_name": Path(resolved_path).name,
                "file_extension": Path(resolved_path).suffix.lower(),
                "mime_type": metadata.get('mimetype', self.get_mime_type(resolved_path)),
                "size_bytes": len(content),
                "size_mb": round(len(content) / (1024 * 1024), 2),
                "modified_time": metadata.get('updated_at', datetime.now().isoformat()),
                "is_file": True,
                "is_directory": False,
                "is_readable": True,
                "is_writable": True,
                "exists": True
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class SupabaseSaveTool(Tool):
    name: str = "supabase_save"
    description: str = "Save content to a file in Supabase remote storage with automatic format detection"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to save in Supabase"
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

    def __init__(self, storage_handler: SupabaseStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler

    def __call__(self, file_path: str, content: str, encoding: str = "utf-8", indent: int = 2, 
                 sheet_name: str = "Sheet1", root_tag: str = "root") -> Dict[str, Any]:
        """
        Save content to a file in Supabase remote storage.
        
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
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    parsed_content = content
            
            # Handle CSV content
            elif file_extension == '.csv':
                try:
                    parsed_content = json.loads(content)
                    if not isinstance(parsed_content, list):
                        return {"success": False, "error": "CSV content must be a list of dictionaries"}
                except json.JSONDecodeError:
                    return {"success": False, "error": "CSV content must be valid JSON array"}
            
            # Handle Excel content
            elif file_extension == '.xlsx':
                try:
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
            logger.error(f"Error in SupabaseSaveTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class SupabaseReadTool(Tool):
    name: str = "supabase_read"
    description: str = "Read content from a file in Supabase remote storage with automatic format detection"
    inputs: Dict[str, Dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read from Supabase"
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

    def __init__(self, storage_handler: SupabaseStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler

    def __call__(self, file_path: str, encoding: str = "utf-8", sheet_name: str = None, head: int = 0) -> Dict[str, Any]:
        """
        Read content from a file in Supabase remote storage.
        
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
            logger.error(f"Error in SupabaseReadTool: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}


class SupabaseDeleteTool(Tool):
    name: str = "supabase_delete"
    description: str = "Delete a file from Supabase remote storage"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to the file to delete from Supabase"
        }
    }
    required: Optional[List[str]] = ["path"]

    def __init__(self, storage_handler: SupabaseStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler

    def __call__(self, path: str) -> Dict[str, Any]:
        """
        Delete a file from Supabase remote storage.
        
        Args:
            path: Path to the file to delete
            
        Returns:
            Dictionary containing the delete operation result
        """
        try:
            result = self.storage_handler.delete(path)
            return result
            
        except Exception as e:
            logger.error(f"Error in SupabaseDeleteTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class SupabaseListTool(Tool):
    name: str = "supabase_list"
    description: str = "List files and directories in Supabase remote storage"
    inputs: Dict[str, Dict[str, str]] = {
        "path": {
            "type": "string",
            "description": "Path to list files from in Supabase (default: root directory)"
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

    def __init__(self, storage_handler: SupabaseStorageHandler = None):
        super().__init__()
        self.storage_handler = storage_handler

    def __call__(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        """
        List files and directories in Supabase remote storage.
        
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
            logger.error(f"Error in SupabaseListTool: {str(e)}")
            return {"success": False, "error": str(e), "path": path}


class StorageSupabaseToolkit(Toolkit):
    """
    Supabase remote storage toolkit.
    Provides tools for reading, writing, deleting, and listing files in Supabase remote storage.
    Requires SUPABASE_URL and SUPABASE_ANON_KEY environment variables to be set.
    """
    
    def __init__(self, name: str = "StorageSupabaseToolkit", bucket_name: str = "default", base_path: str = "/"):
        """
        Initialize the Supabase storage toolkit.
        
        Args:
            name: Name of the toolkit
            bucket_name: Supabase storage bucket name
            base_path: Base path for storage operations
        """
        # Create the shared storage handler instance
        storage_handler = SupabaseStorageHandler(
            bucket_name=bucket_name,
            base_path=base_path
        )
        
        # Create tools with the storage handler
        tools = [
            SupabaseSaveTool(storage_handler=storage_handler),
            SupabaseReadTool(storage_handler=storage_handler),
            SupabaseDeleteTool(storage_handler=storage_handler),
            SupabaseListTool(storage_handler=storage_handler)
        ]
        
        super().__init__(name=name, tools=tools)
        self.storage_handler = storage_handler
