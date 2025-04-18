import unittest
import os
import dotenv
from unittest.mock import patch
from evoagentx.tools.search_google import SearchGoogle

dotenv.load_dotenv()

class TestSearchGoogle(unittest.TestCase):
    def setUp(self):
        self.search_tool = SearchGoogle(num_search_pages=2, max_content_words=100)
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
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

    @patch('evoagentx.tools.search_google.requests.get')
    @patch('evoagentx.tools.search_base.SearchBase._scrape_page')
    def test_search_success(self, mock_scrape_page, mock_get):
        mock_get.return_value.json.return_value = {
            'items': [
                {'title': 'Python', 'link': 'https://www.python.org/'}
            ]
        }
        mock_scrape_page.return_value = ('Python', 'Python is a programming language...')

        # Test both the legacy search method and the new execute method
        search_result = self.search_tool.search('Python', {'api_key': self.api_key, 'search_engine_id': self.search_engine_id})
        execute_result = self.search_tool.execute(query='Python')
        
        # Verify both methods return the same structure
        self.assertEqual(search_result, execute_result)
        
        # Verify the result content
        self.assertIn('results', execute_result)
        self.assertIsNone(execute_result['error'])
        self.assertEqual(len(execute_result['results']), 1)
        self.assertIn('Python', execute_result['results'][0]['title'])

    @patch('evoagentx.tools.search_google.requests.get')
    def test_search_no_results(self, mock_get):
        mock_get.return_value.json.return_value = {}
        
        # Test using the new execute interface
        result = self.search_tool.execute(query='NonExistentQuery')
        self.assertIn('results', result)
        self.assertEqual(result['results'], [])
        self.assertEqual(result['error'], 'No search results found.')

    @patch('evoagentx.tools.search_google.requests.get')
    def test_search_api_error(self, mock_get):
        mock_get.side_effect = Exception("API Error")
        
        # Test using the new execute interface
        result = self.search_tool.execute(query='Python')
        self.assertIn('results', result)
        self.assertEqual(result['results'], [])
        self.assertEqual(result['error'], 'API Error')

if __name__ == '__main__':
    unittest.main()

