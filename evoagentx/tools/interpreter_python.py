import ast
import abc
import contextlib
import io
import importlib
import os
import subprocess
from typing import List, Set, Optional, Union, Any, Dict
from .interpreter_base import BaseInterpreter


class Interpreter_Python(BaseInterpreter):
    ALLOWED_CODE_TYPES = {"python", "py", "python3"}
    
    def __init__(self, project_path: str, allowed_imports: Optional[Set[str]] = None, namespace: Optional[Dict[str, Any]] = None, allowed_functions: Optional[Set[str]] = None):
        """Interpreter that checks the code for safety before execution."""
        # sample project_path: "./AGTChart"
        # sample project_name: "AGTChart"
        
        self.project_path = project_path
        self.project_name = project_path.split("/")[-1]
        self.allowed_imports = allowed_imports if allowed_imports is not None else set()
        self.allowed_functions = allowed_functions if allowed_functions is not None else set()
        self.namespace = namespace or dict()  # Dictionary to store imports and variables

    def _update_namespace(self, alias: str, module_name: Any) -> None:
        self.namespace[alias] = module_name
        

    def _check_init(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        violations = self._analyze_code(code)
        return violations

    def _check_project(self, module: Union[ast.Import, ast.ImportFrom]) -> List[str]:
        if isinstance(module, ast.Import):
            module_name = module.name
        else:
            module_name = module.module

        if len(module_name.split(".")) > 1:
            init_path = self.project_path[:-len(self.project_name)] + "/".join(module_name.split(".")[:-1]) + "/__init__.py"
            module_path = self.project_path[:-len(self.project_name)] + "/".join(module_name.split(".")) + ".py"
        else:
            init_path = self.project_path[:-len(self.project_name)] + "__init__.py"
            module_path = self.project_path[:-len(self.project_name)] + module_name + ".py"
        full_path = self.project_path[:-len(self.project_name)] + "/".join(module_name.split(".")) + "/__init__.py"
        
        if os.path.exists(init_path):
            if init_path in self.visited_modules:
                return []
            self.visited_modules[init_path] = True
            return self._check_init(init_path)
        
        elif os.path.exists(module_path):
            if module_path in self.visited_modules:
                return []
            self.visited_modules[module_path] = True
            with open(module_path, "r", encoding="utf-8") as f:
                code = f.read()
            return self._analyze_code(code)
        
        elif os.path.exists(full_path):
            if full_path in self.visited_modules:
                return []
            self.visited_modules[full_path] = True
            return self._check_init(full_path)
        
        else:
            return [f"Module not found: {module_name}"]

    def _execute_import(self, import_module: ast.Import) -> List[str]:
        violations = []
        for module in import_module.names:
            if module.name.startswith(self.project_name):
                violations += self._check_project(module)
                continue
            if module.name not in self.allowed_imports:
                violations.append(f"Unauthorized import: {module.name}")
                continue
            try:
                alias = module.asname or module.name
                imported_module = importlib.import_module(module.name)
                self._update_namespace(alias, imported_module)
            except ImportError:
                violations.append(f"Failed to import: {module.name}")
        return violations

    def _execute_import_from(self, import_from: ast.ImportFrom) -> List[str]:
        
        if import_from.module is None:
            return ["'from . import' is not supported."]
        if import_from.module.startswith(self.project_name):
            return self._check_project(import_from)
        if import_from.module not in self.allowed_imports:
            return [f"Unauthorized import: {import_from.module}"]

        try:
            for import_name in import_from.names:
                imported_module = importlib.import_module(import_from.module)
                alias = import_name.asname or import_name.name
                self._update_namespace(alias, getattr(imported_module, import_name.name))
            return []
        except ImportError:
            return [f"Failed to import: {import_from.module}"]

    
    def _analyze_code(self, code: str) -> List[str]:
        """Parses and analyzes the code for violations before execution."""
        violations = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    violations += self._execute_import(node)
                elif isinstance(node, ast.ImportFrom):
                    violations += self._execute_import_from(node)
        except SyntaxError as e:
            violations.append(f"Syntax error in code: {e}")
        return violations

    def execute(self, code: str, codetype: str) -> str:
        self.visited_modules = {}
        self.namespace = dict()

        """Checks the code for safety and executes it if there are no violations."""
        if codetype not in self.ALLOWED_CODE_TYPES:
            return f"Unsupported code type: {codetype}. Allowed types are: {', '.join(self.ALLOWED_CODE_TYPES)}"

        violations = self._analyze_code(code)
        
        if violations:
            return "\n".join(violations)

        # Run the code in a subprocess for isolation
        process = subprocess.Popen(
            ["python", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return f"Execution error: {stderr.strip()}"
        
        return stdout.strip()