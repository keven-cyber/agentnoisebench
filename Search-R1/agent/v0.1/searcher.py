"""Searcher module that calls the local retrieval service used in infer.py.

It mirrors the formatting used in the original `infer.py` so the agent can
append results into the prompt in the same structure.
"""
import requests
from typing import List


class Searcher:
    def __init__(self, retrieve_url: str = "http://127.0.0.1:8008/retrieve"):
        self.retrieve_url = retrieve_url

    def _passages2string(self, retrieval_result: List[dict]) -> str:
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item.get("document", {}).get("contents", "")
            title = content.split("\n")[0] if content else ""
            text = "\n".join(content.split("\n")[1:]) if content else ""
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    def search(self, query: str, topk: int = 3) -> str:
        payload = {"queries": [query], "topk": topk, "return_scores": True}
        resp = requests.post(self.retrieve_url, json=payload, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("result")
        if not results:
            return ""
        # return formatted string for the first query
        return self._passages2string(results[0])


# A minimal JSON Schema-like description for the `search` tool. This is
# informational and used by the agent to validate arguments before calling.
SEARCH_TOOL_SCHEMA = {
    "name": "search",
    "description": "Search the local retrieval service and return formatted passages.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string."},
            "topk": {"type": "integer", "description": "Number of results to return.", "default": 3}
        },
        "required": ["query"]
    }
}


def call_search_with_args(searcher: Searcher, args: dict) -> str:
    """Validate args against the minimal schema and call the searcher.

    This function performs light-weight validation (type checks) and then
    calls `searcher.search(query, topk)`.
    """
    if not isinstance(args, dict):
        raise ValueError("search tool arguments must be an object/dict")

    query = args.get("query")
    if not query or not isinstance(query, str):
        raise ValueError("search tool requires a string 'query' field")

    topk = args.get("topk", 3)
    try:
        topk = int(topk)
    except Exception:
        topk = 3

    return searcher.search(query, topk=topk)
