import io
import shlex
import tarfile
import uuid
import docker
from pathlib import Path
from typing import ClassVar, Dict, Any
from .interpreter_base import BaseInterpreter

class DockerInterpreter(BaseInterpreter):
    """
    A Docker-based interpreter for executing Python, Bash, and R scripts in an isolated environment.
    """

    


    _CODE_EXECUTE_CMD_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python {file_name}",
    }

    _CODE_TYPE_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
    }

    require_confirm:bool = False
    print_stdout:bool = True
    print_stderr:bool = True
    _container:docker.models.containers.Container = None
    image_tag:str = "fundingsocietiesdocker/python3.9-slim"
    dockerfile_path:str = "./docker/Dockerfile"
    host_directory:str = ""
    container_directory:str = "/home/app/"
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-pydantic types like sets

    def __init__(self, **data):
        super().__init__(**data)
        self._client = docker.from_env()
        self._initialize_if_needed()
        if self.host_directory:
            self._upload_directory_to_container(self.host_directory)
        

    def __del__(self):
        if self._container is not None:
            self._container.remove(force=True)

    def _initialize_if_needed(self):
        if self._container is not None:
            return
        
        dockerfile_path = Path("./docker/Dockerfile")

        try:
            self._client.images.get(self.image_tag)
        except docker.errors.ImageNotFound:
            if not dockerfile_path.exists():
                raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")
            
            self._client.images.build(path="./docker", tag=self.image_tag, rm=True, buildargs={})

        self._container = self._client.containers.run(
            self.image_tag, detach=True, command="tail -f /dev/null",working_dir="/home/app/"
        )

        # self.execute(, "python")

    def _upload_directory_to_container(self, host_directory: str):
        """
        Uploads all files and directories from the given host directory to /home/app/ in the container.

        :param host_directory: Path to the local directory containing files to upload.
        :param container_directory: Target directory inside the container (defaults to "/home/app/").
        """
        host_directory = Path(host_directory).resolve()
        if not host_directory.exists() or not host_directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {host_directory}")

        tar_stream = io.BytesIO()
        
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for file_path in host_directory.rglob("*"):
                if file_path.is_file():
                    # Ensure path is relative to the given directory
                    relative_path = file_path.relative_to(host_directory)
                    target_path = Path(self.container_directory) / relative_path
                    
                    tarinfo = tarfile.TarInfo(name=str(target_path.relative_to(self.container_directory)))
                    tarinfo.size = file_path.stat().st_size
                    with open(file_path, "rb") as f:
                        tar.addfile(tarinfo, f)

        tar_stream.seek(0)

        if self._container is None:
            raise RuntimeError("Container is not initialized.")

        self._container.put_archive(self.container_directory, tar_stream)

        # Ensure the uploaded directory is in sys.path for imports
        # self._container.exec_run(f"echo 'export PYTHONPATH=/home/app/:$PYTHONPATH' | sudo tee -a /etc/environment")

    def _create_file_in_container(self, content: str) -> Path:
        filename = str(uuid.uuid4())
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content.encode('utf-8')))
        tar_stream.seek(0)

        if self._container is None:
            raise RuntimeError("Container is not initialized.")
        self._container.put_archive("/tmp", tar_stream)
        return Path(f"/tmp/{filename}")

    def _run_file_in_container(self, file: Path, language: str) -> str:
        language = self._check_language(language)
        command = shlex.split(self._CODE_EXECUTE_CMD_MAPPING[language].format(file_name=file.as_posix()))
        if self._container is None:
            raise RuntimeError("Container is not initialized.")
        result = self._container.exec_run(command, demux=True)

        stdout, stderr = result.output
        if self.print_stdout and stdout:
            print(stdout.decode())
        if self.print_stderr and stderr:
            print(stderr.decode())

        return stdout.decode() if stdout else "" + (stderr.decode() if stderr else "")

    def execute(self, code: str, language: str) -> str:
        """
        Executes code in a Docker container.
        
        Args:
            code (str): The code to execute
            language (str): The programming language to use
            
        Returns:
            str: The execution output
        """
        if self.host_directory:
            code = f"import sys; sys.path.insert(0, '{self.container_directory}');" + code
        language = self._check_language(language)
        if self.require_confirm:
            confirmation = input(f"Confirm execution of {language} code? [Y/n]: ")
            if confirmation.lower() not in ["y", "yes", ""]:
                raise RuntimeError("Execution aborted by user.")
        
        
        file_path = self._create_file_in_container(code)
        return self._run_file_in_container(file_path, language)

    def _check_language(self, language: str) -> str:
        if language not in self._CODE_TYPE_MAPPING:
            raise ValueError(f"Unsupported language: {language}")
        return self._CODE_TYPE_MAPPING[language]

    def get_tool_schemas(self) -> list[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for the Docker interpreter.
        
        Returns:
            list[Dict[str, Any]]: Function schema in OpenAI format
        """
        return [{
            "name": "execute",
            "description": "The Docker Interpreter Tool provides a secure and isolated environment for executing code inside a Docker container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to execute"
                    },
                    "language": {
                        "type": "string",
                        "description": "The programming language of the code (e.g., python, py, python3)"
                    }
                },
                "required": ["code", "language"]
            }
        }]
        
    def get_tool_schema(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility. Use get_tool_schemas instead."""
        schemas = self.get_tool_schemas()
        if schemas and len(schemas) > 0:
            return schemas[0]
        return {}

    def get_tool_descriptions(self) -> str:
        """
        Returns a brief description of the Docker interpreter tool.
        
        Returns:
            str: Tool description
        """
        return [
            "Docker Interpreter Tool that provides a secure and isolated environment for executing code inside Docker containers."
        ]
        
    def get_tools(self):
        """
        Returns a list of callable methods provided by this tool.
        
        Returns:
            list: List of callable methods
        """
        return [self.execute]