import unittest
from evoagentx.tools.interpreter_python import Interpreter_Python  # Assuming the interpreter file is named interpreter.py

class TestInterpreterPython(unittest.TestCase):
    
    def setUp(self):
        self.interpreter = Interpreter_Python(
            project_path="./",
            allowed_imports={"math", "random"},
            allowed_functions={"print"}
        )
    
    def test_initialization(self):
        self.assertIn("math", self.interpreter.allowed_imports)
        self.assertIn("print", self.interpreter.allowed_functions)
        self.assertIsInstance(self.interpreter.namespace, dict)
    
    def test_allowed_import(self):
        code = "import math"
        violations = self.interpreter._analyze_code(code)
        self.assertEqual(violations, [])  # No violations expected
    
    def test_unauthorized_import(self):
        code = "import os"
        violations = self.interpreter._analyze_code(code)
        self.assertIn("Unauthorized import: os", violations)
    
    def test_syntax_error(self):
        code = "def test_func("
        violations = self.interpreter._analyze_code(code)
        self.assertTrue(any("Syntax error in code" in v for v in violations))
    
    def test_execution_with_valid_code(self):
        code = "print('Hello World')"
        output = self.interpreter.execute(code, "python")
        self.assertEqual(output, "Hello World")
    
    def test_execution_with_unauthorized_import(self):
        code = "import os\nprint('This should fail')"
        output = self.interpreter.execute(code, "python")
        self.assertIn("Unauthorized import: os", output)
    
    def test_execution_with_syntax_error(self):
        code = "print('Hello"  # Missing closing quote
        output = self.interpreter.execute(code, "python")
        self.assertIn("Syntax error in code", output)
    
    def test_import_from_allowed_module(self):
        code = "from math import sqrt"
        violations = self.interpreter._analyze_code(code)
        self.assertEqual(violations, [])  # No violations expected
    
    def test_import_from_unauthorized_module(self):
        code = "from os import system"
        violations = self.interpreter._analyze_code(code)
        self.assertIn("Unauthorized import: os", violations)
    
    def test_execute_unsupported_code_type(self):
        code = "print('Hello')"
        output = self.interpreter.execute(code, "javascript")
        self.assertIn("Unsupported code type", output)

if __name__ == "__main__":
    unittest.main()
