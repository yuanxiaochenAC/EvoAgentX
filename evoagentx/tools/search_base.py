import requests
from bs4 import BeautifulSoup
from typing import List, Tuple, Callable, Optional
from .tool import Tool
from pydantic import Field

class SearchBase(Tool):
    """
    Base class for search tools that retrieve information from various sources.
    Implements the standard tool interface with get_tool_schemas and execute methods.
    """
    
    num_search_pages: int = Field(default=5, description="Number of search results to retrieve")
    max_content_words: int = Field(default=1000, description="Maximum number of words to include in content. Default None means no limit.")
    
    def __init__(
        self, 
        name: str = "Base Search Tool",
        schemas: Optional[List[dict]] = None,
        descriptions: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        num_search_pages: int = 5, 
        **kwargs
    ):
        """
        Initialize the base search tool.
        
        Args:
            name (str): Name of the tool
            schemas (List[dict], optional): Tool schemas
            descriptions (List[str], optional): Tool descriptions
            tools (List[Any], optional): Tool functions
            num_search_pages (int): Number of search results to retrieve
            max_content_words (int): Maximum number of words to include in content, default None means no limit. 
            **kwargs: Additional keyword arguments for parent class initialization
        """
        # Set default values if not provided
        schemas = schemas or self.get_tool_schemas()
        descriptions = descriptions or self.get_tool_descriptions()
        tools = tools or self.get_tools()
            
        # Pass to parent class initialization
        super().__init__(
            name=name,
            schemas=schemas,
            descriptions=descriptions,
            tools=tools,
            **kwargs
        )

        # Override default values if provided
        self.num_search_pages = num_search_pages
    
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
        paragraph_texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        text_content = "\n\n".join(paragraph_texts)

        return title, text_content