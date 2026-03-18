"""Web search tool - DuckDuckGo integration"""

from src.tools.base import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web via DuckDuckGo. Returns titles, URLs, snippets."
    parameters = {
        "query": {"description": "search query", "required": True},
        "max_results": {"description": "number of results, default 5", "required": False},
    }

    def execute(self, **kwargs) -> str:
        query = kwargs.get("query", "")
        max_results = int(kwargs.get("max_results", 5))

        if not query:
            return "Error: please provide a search query"

        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return f"No results found for: {query}"

            output = []
            for i, r in enumerate(results, 1):
                output.append(
                    f"{i}. {r.get('title', 'N/A')}\n"
                    f"   URL: {r.get('href', 'N/A')}\n"
                    f"   {r.get('body', 'N/A')}"
                )
            return "\n\n".join(output)

        except Exception as e:
            return f"Search error: {e}"
