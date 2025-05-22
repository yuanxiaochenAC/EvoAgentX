import requests
import os
from typing import Dict, Any, Optional
from .search_base import SearchBase
from evoagentx.core.logging import logger
import dotenv
dotenv.load_dotenv()

class SearchGoogle(SearchBase):

    def __init__(
        self, 
        name: str = 'SearchGoogle',
        num_search_pages: Optional[int] = 5, 
        max_content_words: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the Google Search tool.
        
        Args:
            name (str): The name of the search tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int, optional): Maximum number of words to include in content, None means no limit
            **kwargs: Additional data to pass to the parent class
        """
        # Pass these to the parent class initialization
        super().__init__(name=name, num_search_pages=num_search_pages, max_content_words=max_content_words, **kwargs)
    
    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None) -> Dict[str, Any]:
        """
        Search Google using the Custom Search API and retrieve detailed search results with content snippets.
        
        Args:
            query (str): The search query to execute on Google
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, None means no limit
            
        Returns:
            Dict[str, Any]: Contains search results and optional error message
        """
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words
        results = []
        
        # Get API credentials from environment variables
        api_key = os.getenv('GOOGLE_API_KEY', '')
        search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
        
        # print(f"api_key: {api_key}")
        # print(f"search_engine_id: {search_engine_id}")
            
        if not api_key or not search_engine_id:
            error_msg = (
                "API key and search engine ID are required. "
                "Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables. "
                "You can get these from the Google Cloud Console: https://console.cloud.google.com/apis/"
            )
            logger.error(error_msg)
            return {"results": [], "error": error_msg}
        
        try:
            # Step 1: Query Google Custom Search API
            logger.info(f"Searching Google for: {query}, num_results={num_search_pages}, max_content_words={max_content_words}")
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": search_engine_id,
                "q": query,
                "num": num_search_pages,
            }
            response = requests.get(search_url, params=params)
            data = response.json()

            if "items" not in data:
                return {"results": [], "error": "No search results found."}

            search_results = data["items"]
            logger.info(f"Found {len(search_results)} search results")

            # Step 2: Fetch content from each valid search result
            for item in search_results:
                url = item["link"]
                title = item["title"]
                try:
                    title, content = self._scrape_page(url)
                    if content:  # Ensure valid content exists
                        # Use the base class's content truncation method
                        display_content = self._truncate_content(content, max_content_words)
                        
                        results.append({
                            "title": title,
                            "content": display_content,
                            "url": url,
                        })
                except Exception as e:
                    logger.warning(f"Error processing URL {url}: {str(e)}")
                    continue  # Skip pages that cannot be processed

            return {"results": results, "error": None}

        except Exception as e:
            logger.error(f"Error searching Google: {str(e)}")
            return {"results": [], "error": str(e)}

    def get_tool_schemas(self) -> list[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for the Google search tool.
        
        Returns:
            list[Dict[str, Any]]: Function schema in OpenAI format
        """
        return [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search Google and retrieve content from search results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute on Google"
                        },
                        "num_search_pages": {
                            "type": "integer",
                            "description": "Number of search results to retrieve. Default: 5"
                        },
                        "max_content_words": {
                            "type": "integer",
                            "description": "Maximum number of words to include in content per result. None means no limit. Default: None"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]
    
    def get_tools(self):
        return [self.search]
        
    def get_tool_descriptions(self) -> list[str]:
        """
        Returns a brief description of the Google search tool.
        
        Returns:
            list[str]: Tool descriptions
        """
        return ["Search Google and retrieve content from search results."]
