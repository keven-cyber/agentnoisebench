"""Simple wrapper for a commercial LLM HTTP API.

This is intentionally generic: it reads API key and base URL from environment
variables by default but allows overriding via constructor.

Expected env vars:
- OPENAI_API_KEY
- OPENAI_BASE_URL

The client will POST JSON {"prompt": ..., "max_tokens": ..., "temperature": ...}
and try to extract the generated text from common response shapes.
"""
import os
import requests
from typing import Optional, Dict, Any, List


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, default_kwargs: Optional[Dict[str, Any]] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.default_kwargs = default_kwargs or {"max_tokens": 512, "temperature": 0.7}

    def generate(self, prompt: str, functions: Optional[List[dict]] = None, **kwargs) -> str:
        """Send prompt to the LLM API and return the text output.

        You can optionally provide `functions` (a list of tool/function schemas) and
        they will be included verbatim in the JSON payload under the `functions` key.

        The method is tolerant to a few common JSON response shapes and falls back
        to returning the raw response text when no known field is found.
        """
        payload = {**self.default_kwargs, **kwargs}
        payload["prompt"] = prompt
        if functions is not None:
            # include function/tool schemas so the model can perform function/tool calls
            payload["functions"] = functions

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.base_url.rstrip("/") + "/v1/generate"
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # try common locations
        if isinstance(data, dict):
            # openai-like choices.text
            choices = data.get("choices")
            if choices and isinstance(choices, list):
                first = choices[0]
                if isinstance(first, dict):
                    if "text" in first:
                        return first["text"]
                    # chat-like
                    msg = first.get("message") or first.get("delta")
                    if msg and isinstance(msg, dict) and "content" in msg:
                        return msg["content"]

            # direct output
            if "output" in data and isinstance(data["output"], str):
                return data["output"]

            # Cohere-like: data: [{"text": ...}]
            arr = data.get("data")
            if arr and isinstance(arr, list) and "text" in arr[0]:
                return arr[0]["text"]

        # fallback to raw text
        try:
            return resp.text
        except Exception:
            return ""
