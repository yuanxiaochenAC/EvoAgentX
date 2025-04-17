import unittest
from evoagentx.tools.tool import Tool

class TestTool(unittest.TestCase):
    """Tests for the base Tool class"""
    
    def test_abstract_methods(self):
        """Test that Tool abstract methods raise NotImplementedError when not implemented"""
        tool = Tool()
        
        # Test get_tool_schema
        with self.assertRaises(NotImplementedError):
            tool.get_tool_schema()
            
        # Test execute
        with self.assertRaises(NotImplementedError):
            tool.execute(param="test")
    
    def test_legacy_get_tool_info(self):
        """Test that the legacy get_tool_info method calls get_tool_schema"""
        # Create a mock tool class that implements get_tool_schema
        class MockTool(Tool):
            def get_tool_schema(self):
                return {"test": "schema"}
                
            def execute(self, **kwargs):
                return "executed"
        
        tool = MockTool()
        # Test that get_tool_info returns the same as get_tool_schema
        self.assertEqual(tool.get_tool_info(), tool.get_tool_schema())

if __name__ == '__main__':
    unittest.main() 