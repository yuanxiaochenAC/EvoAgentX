import unittest
from evoagentx.tools.interpreter_docker import DockerInterpreter
import os
import tempfile

class TestDockerInterpreter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the project
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = os.path.join(self.temp_dir.name, "AGTChart")
        
        # Create the project directory structure
        os.makedirs(os.path.join(self.project_path, "test"), exist_ok=True)

        # Create evi.py
        with open(os.path.join(self.project_path, "evi.py"), "w") as f:
            f.write("""def func1():\n    return \"Hello from func1\"""")

        # Create test/other.py
        with open(os.path.join(self.project_path, "test", "other.py"), "w") as f:
            f.write("""\n\ndef func2():\n  return \"Hello from func2\"""")

        # Initialize the DockerInterpreter
        self.interpreter = DockerInterpreter(
            require_confirm=False,
            print_stdout=True,
            host_directory=self.project_path,
            container_directory="/home/app/"
        )

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()
        
    def test_get_tool_schema(self):
        """Test the get_tool_schema method returns the correct schema"""
        schema = self.interpreter.get_tool_schema()
        
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "function")
        self.assertIn("function", schema)
        self.assertEqual(schema["function"]["name"], "execute_code")
        
        # Check parameters
        params = schema["function"]["parameters"]
        self.assertIn("properties", params)
        self.assertIn("code", params["properties"])
        self.assertIn("language", params["properties"])
        self.assertIn("required", params)
        self.assertIn("code", params["required"])
        self.assertIn("language", params["required"])

    def test_execute_valid_code(self):
        # Sample code that imports from AGTChart
        test_code = """
import os

print('Hello from Docker!')
"""
        # Using the new execute interface
        result = self.interpreter.execute(code=test_code, language="python")
        self.assertIn("Hello from Docker!", result)

    def test_syntax_error(self):
        test_code = "print('Hello World'"
        result = self.interpreter.execute(code=test_code, language="python")
        self.assertIn("SyntaxError", result)

    def test_runtime_error(self):
        test_code = "print(1/0)"
        result = self.interpreter.execute(code=test_code, language="python")
        self.assertIn("ZeroDivisionError", result)

if __name__ == '__main__':
    unittest.main()