import unittest
from unittest.mock import patch
from evoagentx.tools.search_wiki import SearchWiki

class TestSearchWiki(unittest.TestCase):
    def setUp(self):
        self.search_tool = SearchWiki(num_search_pages=2, max_content_words=100)

    def test_get_tool_schema(self):
        """Test the get_tool_schema method returns the correct schema"""
        schema = self.search_tool.get_tool_schema()
        
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "function")
        self.assertIn("function", schema)
        self.assertEqual(schema["function"]["name"], "search")
        
        # Check parameters
        params = schema["function"]["parameters"]
        self.assertIn("properties", params)
        self.assertIn("query", params["properties"])
        self.assertIn("required", params)
        self.assertIn("query", params["required"])

    @patch('evoagentx.tools.search_wiki.wikipedia.search')
    @patch('evoagentx.tools.search_wiki.wikipedia.page')
    @patch('evoagentx.tools.search_wiki.wikipedia.summary')
    def test_execute_success(self, mock_summary, mock_page, mock_search):
        mock_search.return_value = ['Python (programming language)']
        mock_page.return_value.title = 'Python (programming language)'
        mock_page.return_value.content = 'Python is a programming language...'
        mock_page.return_value.url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
        mock_summary.return_value = 'Python is a programming language...'

        # Test using the execute method
        result = self.search_tool.execute(query='Python')
        self.assertIn('results', result)
        self.assertEqual(len(result['results']), 1)
        self.assertIn('Python (programming language)', result['results'][0]['title'])

    @patch('evoagentx.tools.search_wiki.wikipedia.search')
    def test_execute_no_results(self, mock_search):
        mock_search.return_value = []
        
        # Test using the execute method
        result = self.search_tool.execute(query='NonExistentQuery')
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No search results found.')

if __name__ == '__main__':
    unittest.main()
