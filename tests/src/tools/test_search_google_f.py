import unittest
from unittest.mock import patch
from evoagentx.tools.search_google_f import SearchGoogleFree

class TestSearchGoogleFree(unittest.TestCase):
    def setUp(self):
        self.search_tool = SearchGoogleFree(num_search_pages=2, max_content_words=100)
        
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

    @patch('evoagentx.tools.search_google_f.google_f_search')
    @patch('evoagentx.tools.search_base.SearchBase._scrape_page')
    def test_execute_with_mock(self, mock_scrape_page, mock_search):
        # Mock the google search function to return URLs
        mock_search.return_value = ['https://www.python.org/']
        # Mock the scrape page function to return title and content
        mock_scrape_page.return_value = ('Python', 'Python is a programming language...')
        
        # Test the execute method
        result = self.search_tool.execute(query='Python')
        
        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
        self.assertIsInstance(result['results'], list)
        self.assertGreater(len(result['results']), 0)
        
        # Check content of results
        self.assertEqual(result['results'][0]['title'], 'Python')
        self.assertIn('Python is a programming language', result['results'][0]['content'])
        self.assertEqual(result['results'][0]['url'], 'https://www.python.org/')

    @patch('evoagentx.tools.search_google_f.google_f_search')
    def test_search_with_no_results(self, mock_search):
        # Mock empty search results
        mock_search.return_value = []
        
        # Test both the search method and execute method
        result = self.search_tool.execute(query='NonExistentQuery')
        
        # Verify the result
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No search results found.')
        self.assertEqual(result['results'], [])
        
    def test_real_search(self):
        """Test with real Google search results instead of mocks"""
        # Use a query that should return stable results
        query = "Python programming language official website"
        result = self.search_tool.execute(query=query)
        
        # Check the structure of the result
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
        self.assertIsNone(result.get('error'))
        
        # Verify we got results back
        self.assertIsInstance(result['results'], list)
        self.assertGreater(len(result['results']), 0)
        
        # Check that each result has the expected structure
        for item in result['results']:
            self.assertIn('title', item)
            self.assertIn('content', item)
            self.assertIn('url', item)
            
            # Check that the content has been properly truncated
            self.assertLessEqual(len(item['content'].split()), 
                               self.search_tool.max_content_words + 1)  # +1 for the "..."

if __name__ == '__main__':
    unittest.main()
