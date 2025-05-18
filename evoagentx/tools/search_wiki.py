import wikipedia
from .search_base import SearchBase
from typing import Dict, Any
from pydantic import Field
from evoagentx.core.logging import logger


class SearchWiki(SearchBase):
    max_sentences: int = Field(default=10, description="Maximum number of sentences in the summary. Default 0 means return all available content.")
    
    def __init__(
        self, 
        name: str = 'Wikipedia Search',
        num_search_pages: int = 5, 
        max_sentences: int = 50,
        **kwargs
    ):
        """
        Initialize the Wikipedia Search tool.
        
        Args:
            name (str): The name of the search tool
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int, optional): Maximum number of words to include in content, None means no limit
            max_sentences (int): Maximum number of sentences in the summary
            **kwargs: Additional data to pass to the parent class
        """
        schemas = kwargs.pop('schemas', None) or self.get_tool_schemas()
        descriptions = kwargs.pop('descriptions', None) or self.get_tool_descriptions()
        tools = kwargs.pop('tools', None)
        tools = self.get_tools()
        # Pass these to the parent class initialization
        super().__init__(
            name=name,
            schemas=schemas,
            descriptions=descriptions,
            tools=tools,
            num_search_pages=num_search_pages,
            **kwargs
        )
        self.max_sentences = max_sentences

    def search(self, query: str, num_search_pages: int = None, max_content_words: int = None, max_sentences: int = None) -> Dict[str, Any]:
        """
        Searches Wikipedia for the given query and returns the summary and truncated full content.

        Args:
            query (str): The search query.
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, None means no limit
            max_sentences (int): Maximum number of sentences in the summary

        Returns:
            dict: A dictionary with the title, summary, truncated content, and Wikipedia page link.
        """
        if num_search_pages is None:
            num_search_pages = self.num_search_pages
        if max_sentences is None:
            max_sentences = self.max_sentences
            
        try:
            logger.info(f"Searching wikipedia: {query}, max_sentences={max_sentences}, num_results={num_search_pages}")
            # Search for top matching titles
            search_results = wikipedia.search(query, results=num_search_pages)
            logger.info(f"Search results: {search_results}")
            if not search_results:
                return {"results": [], "error": "No search results found."}

            # Try fetching the best available page
            results = []
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Handle the max_sentences parameter
                    if max_sentences is not None and max_sentences > 0:
                        summary = wikipedia.summary(title, sentences=max_sentences)
                    else:
                        # Get the full summary without limiting sentences
                        summary = wikipedia.summary(title)

                    # Truncate content if needed and add ellipsis only if truncated
                    if max_content_words is not None and max_content_words > 0:
                        # This preserves the original spacing while limiting word count
                        words = page.content.split()
                        is_truncated = len(words) > max_content_words
                        word_count = 0
                        truncated_content = ""
                        
                        # Rebuild the content preserving original whitespace
                        for i, char in enumerate(page.content):
                            if char.isspace():
                                if i > 0 and not page.content[i-1].isspace():
                                    word_count += 1
                                if word_count >= max_content_words:
                                    break
                            truncated_content += char
                            
                        # Add ellipsis only if truncated
                        display_content = truncated_content
                        if is_truncated:
                            display_content += " ..."
                    else:
                        # Use full content if max_content_words is None or <= 0
                        display_content = page.content
                    
                    
                    results.append({
                        "title": page.title,
                        "summary": summary,
                        "content": display_content,
                        "url": page.url,
                    })
                except wikipedia.exceptions.DisambiguationError:
                    # Skip ambiguous results and try the next
                    continue
                except wikipedia.exceptions.PageError:
                    # Skip non-existing pages and try the next
                    continue
            
            logger.info(f"get results from wikipedia: {results}")
            return {"results": results}
        
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {str(e)}")
            return {"results": [], "error": str(e)}
    
    def get_tools(self):
        return [self.search]

    def get_tool_schemas(self) -> list[Dict[str, Any]]:
        """
        Returns the OpenAI-compatible function schema for the Wikipedia search tool.
        
        Returns:
            list[Dict[str, Any]]: Function schema in OpenAI format
        """
        return [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search Wikipedia for relevant articles and content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on Wikipedia"
                        },
                        "num_search_pages": {
                            "type": "integer",
                            "description": "Number of search results to retrieve. Default: 5"
                        },
                        "max_content_words": {
                            "type": "integer",
                            "description": "Maximum number of words to include in content per result. None means no limit. Default: None."
                        },
                        "max_sentences": {
                            "type": "integer",
                            "description": "Maximum number of sentences in the summary. Default: 50"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]
        
    def get_tool_descriptions(self) -> list[str]:
        """
        Returns a brief description of the Wikipedia search tool.
        
        Returns:
            list[str]: Tool descriptions
        """
        return [
            "Search Wikipedia for relevant articles and content."
        ]

