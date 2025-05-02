import requests
import os
from typing import Dict, Any
from .search_base import SearchBase

class SearchGoogle(SearchBase):
    
    def get_tool_description(self) -> str:
        """
        Returns a brief description of the Google search tool.
        
        Returns:
            str: Tool description
        """
        return "Google Search Tool that utilizes the Google Custom Search API to perform structured search queries."

    
    def search(self, query: str, api_key = None, search_engine_id = None, num_search_pages:int = 5, max_content_words:int = 500) -> dict:
        results = []
        
        if not api_key :
            api_key = os.getenv('GOOGLE_API_KEY', '')
        if not search_engine_id:
            search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
            
        if not api_key or not search_engine_id:
            raise ValueError("API key and search engine ID are required.")
        
        try:
            # Step 1: Query Google Custom Search API
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

            # Step 2: Fetch content from each valid search result
            for item in search_results:
                url = item["link"]
                title = item["title"]
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
        Returns the OpenAI-compatible function schema for the Google search tool.
        
        Returns:
            Dict[str, Any]: Function schema in OpenAI format
        """
        return [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search Google using the Custom Search API and retrieve detailed search results with content snippets.",
                "parameters": {
                    "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute on Google"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "The Google API key to use (will use environment variable if not provided)"
                    },
                    "search_engine_id": {
                        "type": "string",
                        "description": "The Google Search Engine ID to use (will use environment variable if not provided)"
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
    
    def get_tools(self):
        return [self.search]
    
    def get_tool_descriptions(self) -> str:
        return [
            "Google Search Tool that utilizes the Google Custom Search API to perform structured search queries."
        ]
