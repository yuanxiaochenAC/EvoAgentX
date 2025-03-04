import io
import shlex
import tarfile
import uuid
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import docker
from docker.models.containers import Container

class DockerInterpreter:
    """
    A Docker-based interpreter for executing Python, Bash, and R scripts in an isolated environment.
    """
    _CODE_EXECUTE_CMD_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python {file_name}",
    }

    _CODE_EXTENSION_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "py",
    }

    _CODE_TYPE_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
    }

    def __init__(self, require_confirm: bool = True, print_stdout: bool = False, print_stderr: bool = True, image_tag: str = "fundingsocietiesdocker/python3.9-slim", dockerfile_path: str = "./docker/Dockerfile"):
        self.require_confirm = require_confirm
        self.print_stdout = print_stdout
        self.print_stderr = print_stderr
        self._container: Optional[Container] = None
        self._client = docker.from_env()
        self.image_tag = image_tag
        self.dockerfile_path = dockerfile_path

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
            print("Docker image not found. Building the image from ./docker/Dockerfile...")
            if not dockerfile_path.exists():
                raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")
            
            self._client.images.build(path="./docker", tag=self.image_tag, rm=True, buildargs={})

        self._container = self._client.containers.run(
            self.image_tag, detach=True, command="tail -f /dev/null"
        )

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

    def _run_file_in_container(self, file: Path, code_type: str) -> str:
        code_type = self._check_code_type(code_type)
        command = shlex.split(self._CODE_EXECUTE_CMD_MAPPING[code_type].format(file_name=file.as_posix()))
        if self._container is None:
            raise RuntimeError("Container is not initialized.")
        result = self._container.exec_run(command, demux=True)

        stdout, stderr = result.output
        if self.print_stdout and stdout:
            print(stdout.decode())
        if self.print_stderr and stderr:
            print(stderr.decode())

        return stdout.decode() if stdout else "" + (stderr.decode() if stderr else "")

    def run(self, code: str, code_type: str) -> str:
        code_type = self._check_code_type(code_type)
        if self.require_confirm:
            confirmation = input(f"Confirm execution of {code_type} code? [Y/n]: ")
            if confirmation.lower() not in ["y", "yes", ""]:
                raise RuntimeError("Execution aborted by user.")
        
        self._initialize_if_needed()
        file_path = self._create_file_in_container(code)
        return self._run_file_in_container(file_path, code_type)

    def _check_code_type(self, code_type: str) -> str:
        if code_type not in self._CODE_TYPE_MAPPING:
            raise ValueError(f"Unsupported code type: {code_type}")
        return self._CODE_TYPE_MAPPING[code_type]

    def supported_code_types(self) -> List[str]:
        return list(self._CODE_EXTENSION_MAPPING.keys())
