from .search_base import SearchBase
from googlesearch import search as google_f_search
from typing import Dict, Any

class SearchGoogleFree(SearchBase):

    

    
    def search(self, query: str, num_search_pages:int = 5, max_content_words:int = 500) -> Dict[str, Any]:
        """
        Searches Google for the given query and retrieves content from multiple pages.

        Args:
            query (str): The search query.

        Returns:
            Dict[str, Any]: Contains a list of search results and optional error message.
        """
        results = []
        try:
            # Step 1: Get top search result links
            search_results = list(google_f_search(query, num_results=num_search_pages))
            if not search_results:
                return {"results": [], "error": "No search results found."}

            # Step 2: Fetch content from each page
            for url in search_results:
                try:
                    title, content = self._scrape_page(url)
                    if content:  # Ensure valid content exists
                        results.append({
                            "title": title,
                            "content": ' '.join(content.split()[:max_content_words]) + " ...",
                            "url": url,
                        })
                except Exception:
                    continue  # Skip pages that cannot be processed

            return {"results": results, "error": None}
        
        except Exception as e:
            return {"results": [], "error": str(e)}
    
    def get_tool_schemas(self) -> list[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for the free Google search tool.
        
        Returns:
            Dict[str, Any]: Function schema in OpenAI format
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
                        "description": "The number of search results to retrieve (default is 5)."
                    },
                    "max_content_words": {
                        "type": "integer",
                        "description": "The maximum number of words to retain from the content (default is 500)."
                    }
                },
                "required": ["query"]
                }
            }
        }]
        
    def get_tool_descriptions(self) -> str:
        """
        Returns a brief description of the free Google search tool.
        
        Returns:
            str: Tool description
        """
        return [
            "Free Google Search Tool that queries Google without requiring an API key."
        ]
        
    def get_tools(self):
        return [self.search]
