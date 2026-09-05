from typing import List

import requests

class Searcher:
    """
    一个简单的检索封装。暂时只实现了对本地检索服务的调用。
    默认端口 8008。
    """
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
        return self._passages2string(results[0])