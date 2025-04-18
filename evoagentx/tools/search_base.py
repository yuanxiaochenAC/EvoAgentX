import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Tuple
from .tool import Tool

class SearchBase(Tool):
    """
    Base class for search tools that retrieve information from various sources.
    Implements the standard tool interface with get_tool_schema and execute methods.
    """
    
    num_search_pages: int = 5
    max_content_words: int = 500

    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Returns the OpenAI-compatible function schema for the search tool.
        
        Returns:
            Dict[str, Any]: Function schema in OpenAI format
        """
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": self.get_tool_description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def get_tool_description(self) -> str:
        """
        Returns a brief description of the search tool.
        
        Returns:
            str: Tool description
        """
        return "Search tool that retrieves information from various sources based on a query."
    
    def execute(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes a search with the given query.
        
        Args:
            query (str): The search query
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        return self.search(query)
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs the search operation. Must be implemented by subclasses.
        
        Args:
            query (str): The search query
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        raise NotImplementedError("Subclasses must implement search")

    def _scrape_page(self, url: str) -> Tuple[str, str]:
        """
        Fetches the title and main text content from a web page.

        Args:
            url (str): The URL of the web page.

        Returns:
            tuple: (title, main textual content)
        """
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            return None, None

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = soup.title.string if soup.title else "No Title"

        # Extract text content (only from <p> tags)
        paragraphs = soup.find_all("p")
        text_content = " ".join([p.get_text(strip=True) for p in paragraphs])

        return title, text_content