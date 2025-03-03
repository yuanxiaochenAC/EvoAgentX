import abc
import wikipedia
import requests
from bs4 import BeautifulSoup
from googlesearch import search as google_f_search

class Search_Tool(abc.ABC):
    def __init__(self, num_search_pages: int = 5, max_content_words: int = 500):
        """
        Initializes the search tool.

        Args:
            num_search_pages (int): Number of search results to check.
            max_content_words (int): Maximum words for the truncated content.
        """
        self.num_search_pages = num_search_pages
        self.max_content_words = max_content_words

    def search_wikipedia(self, query: str, max_sentences: int = 15) -> list:
        """
        Searches Wikipedia for the given query and returns the summary and truncated full content.

        Args:
            query (str): The search query.
            max_sentences (int): Maximum number of sentences in the summary.

        Returns:
            dict: A dictionary with the title, summary, truncated content, and Wikipedia page link.
        """
        try:
            # Search for top matching titles
            search_results = wikipedia.search(query, results=self.num_search_pages)
            if not search_results:
                return {"error": "No search results found."}

            # Try fetching the best available page
            results = []
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(title, sentences=max_sentences)

                    # Truncate the full content to the first max_content_words words
                    content = ' '.join(page.content.split()[:self.max_content_words])

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

            return results
        
        except Exception as e:
            return {"error": str(e)}
    
    def search_google(self, query: str, api_key: str, search_engine_id: str) -> dict:
        results = []
        try:
            # Step 1: Query Google Custom Search API
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": search_engine_id,
                "q": query,
                "num": self.num_search_pages,
            }

            response = requests.get(search_url, params=params)
            data = response.json()

            if "items" not in data:
                return {"error": "No search results found."}

            search_results = data["items"]

            # Step 2: Fetch content from each valid search result
            for item in search_results:
                url = item["link"]
                title = item["title"]

                try:
                    _, content = self._scrape_page(url)
                    if content:
                        results.append({
                            "title": title,
                            "truncated_content": ' '.join(content.split()[:self.max_content_words]) + " ...",
                            "url": url,
                        })
                except Exception:
                    continue  # Skip pages that cannot be processed

            return {"results": results} if results else {"error": "All retrieved pages were inaccessible or empty."}

        except Exception as e:
            return {"error": str(e)}


    def search_google_f(self, query: str) -> list:
        """
        Searches Google for the given query and retrieves content from multiple pages.

        Args:
            query (str): The search query.

        Returns:
            dict: Contains a list of search results (title, truncated content, and source URL).
        """
        results = []
        try:
            # Step 1: Get top search result links
            search_results = list(google_f_search(query, num_results=self.num_search_pages))
            if not search_results:
                return {"error": "No search results found."}

            # Step 2: Fetch content from each page
            for url in search_results:
                try:
                    title, content = self._scrape_page(url)
                    if content:  # Ensure valid content exists
                        results.append({
                            "title": title,
                            "content": ' '.join(content.split()[:self.max_content_words]) + " ...",
                            "url": url,
                        })
                except Exception:
                    continue  # Skip pages that cannot be processed

            return results
        
        except Exception as e:
            return {"error": str(e)}

    def _scrape_page(self, url: str) -> tuple:
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