from .search_base import SearchBase
from googlesearch import search as google_f_search
from typing import Dict, Any, List, Callable, Optional
from evoagentx.core.logging import logger

class SearchGoogleFree(SearchBase):
    """
    Free Google Search tool that doesn't require API keys.
    """
    
    def __init__(
        self, 
        name: str = "GoogleFreeSearch",
        num_search_pages: Optional[int] = 5, 
        max_content_words: Optional[int] = None,
       **kwargs 
    ):
        """
        Initialize the Free Google Search tool.
        
        Args:
            name (str): Name of the tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(name=name, num_search_pages=num_search_pages, max_content_words=max_content_words, **kwargs)

    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None) -> Dict[str, Any]:
        """
        Searches Google for the given query and retrieves content from multiple pages.

        Args:
            query (str): The search query.
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, None means no limit

        Returns:
            Dict[str, Any]: Contains a list of search results and optional error message.
        """
        # Use class defaults
        num_search_pages = num_search_pages or self.num_search_pages
        max_content_words = max_content_words or self.max_content_words 
            
        results = []
        try:
            # Step 1: Get top search result links
            logger.info(f"Searching Google (Free) for: {query}, num_results={num_search_pages}, max_content_words={max_content_words}")
            search_results = list(google_f_search(query, num_results=num_search_pages))
            if not search_results:
                return {"results": [], "error": "No search results found."}

            logger.info(f"Found {len(search_results)} search results")
            
            # Step 2: Fetch content from each page
            for url in search_results:
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
            logger.error(f"Error in free Google search: {str(e)}")
            return {"results": [], "error": str(e)}
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for the free Google search tool.
        
        Returns:
            list[Dict[str, Any]]: Function schema in OpenAI format
        """
        return [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search Google without requiring an API key and retrieve content from search results.",
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

    def get_tools(self) -> List[Callable]:
        return [self.search]

    def get_tool_descriptions(self) -> List[str]:
        return [
            "Free Google Search Tool that queries Google without requiring an API key."
        ]

