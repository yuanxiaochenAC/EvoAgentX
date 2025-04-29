import wikipedia
from .search_base import SearchBase
from typing import Dict, Any


class SearchWiki(SearchBase):

    def search(self, query: str, max_sentences: int = 15, num_search_pages: int = 5, max_content_words: int = 500) -> list:
        """
        Searches Wikipedia for the given query and returns the summary and truncated full content.

        Args:
            query (str): The search query.
            max_sentences (int): Maximum number of sentences in the summary.

        Returns:
            dict: A dictionary with the title, summary, truncated content, and Wikipedia page link.
        """
        try:
            print("searching wikipedia: ", query, max_sentences, num_search_pages, max_content_words)
            # Search for top matching titles
            search_results = wikipedia.search(query, results=num_search_pages)
            print("search_results: ", search_results)
            if not search_results:
                return {"error": "No search results found."}

            # Try fetching the best available page
            results = []
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(title, sentences=max_sentences)

                    # Truncate the full content to the first max_content_words words
                    content = ' '.join(page.content.split()[:max_content_words])

                    results.append({
                        "title": page.title,
                        "summary": summary,
                        "content": content + " ...",
                        "url": page.url,
                    })
                except wikipedia.exceptions.DisambiguationError:
                    # Skip ambiguous results and try the next
                    continue
                except wikipedia.exceptions.PageError:
                    # Skip non-existing pages and try the next
                    continue
            
            print("get results from wikipedia: ", results)
            return {"results": results}
        
        except Exception as e:
            return {"error": str(e)}
    
    def get_tools(self):
        return [self.search]

    def get_tool_schemas(self) -> list[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for the Wikipedia search tool.
        
        Returns:
            Dict[str, Any]: Function schema in OpenAI format
        """
        return [{
            "name": "search",
            "description": "Search Wikipedia for relevant articles and extract key information including titles, summaries, and content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on Wikipedia."
                    },
                    "num_search_pages": {
                        "type": "integer",
                        "description": "The number of Wikipedia search results to retrieve (default is 5)."
                    },
                    "max_content_words": {
                        "type": "integer",
                        "description": "The maximum number of words to retain from the full Wikipedia page content (default is 500)."
                    },
                    "max_sentences": {
                        "type": "integer",
                        "description": "The maximum number of sentences in the summary returned from Wikipedia (default is 15)."
                    }
                },
                "required": ["query"]
            }
        }]
        
    def get_tool_description(self) -> str:
        return "Search Wikipedia for relevant articles and extract key information including titles, summaries, and content."

