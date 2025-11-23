from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_community.tools.tavily_search import TavilySearchResults
import os

load_dotenv()

#The freshness_hours parameter in your method specifies how recent the search results should be, in hours. The Tavily API typically supports values like 1, 3, 6, 12, 24, 48, and 168 (up to 7 days).
# You can set freshness_hours to any of these values to control the recency of the results

mcp = FastMCP("WebCrawlerMCP", "1.0.0", "A server to crawl web pages and extract data using Tavli API.")

@mcp.tool()
def crawl_web_page(query: str, freshness_hours: int = 168) -> dict:
    """
    Crawl a web page or perform a search using the Tavily API.
    Args:
        query (str): The search query or URL to crawl.
    Returns:
        dict: The search results or extracted content.
    """
    try:

        search = TavilySearchResults(max_results=2)
        return search.invoke({
            "query": query,
            "freshness": f"{freshness_hours}h"
        })
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")
