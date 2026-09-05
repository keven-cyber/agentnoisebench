"""A minimal SearchAgent that follows the infer.py loop but uses a commercial LLM API.

Behavior:
- Construct a prompt similar to infer.py
- Send prompt to configured LLM (via LLMClient)
- If model output contains a <search>...</search> tag, call the Searcher
  and append the results inside <information>...</information>, then continue
- If model output contains <answer>...</answer>, return the answer

This is intentionally small and dependency-free (uses requests only).
"""
from typing import Optional
import re
import json

from .llm import LLMClient
from .searcher import Searcher, call_search_with_args, SEARCH_TOOL_SCHEMA


class SearchAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None, searcher: Optional[Searcher] = None, max_rounds: int = 5):
        self.llm = llm_client or LLMClient()
        self.searcher = searcher or Searcher()
        self.max_rounds = max_rounds

        # template used when adding retrieved information back into the prompt
        self.curr_search_template = "\n\n{output_text}<information>{search_results}</information>\n\n"

    def _extract_tool_call(self, text: str) -> Optional[dict]:
        """Attempt to extract a tool call in JSON form from the model output.

        Accepts a few common shapes that resemble OpenAI function/tool calls:
        - {"name": "search", "arguments": {...}}
        - {"function_call": {"name": "search", "arguments": "{...}"}}

        Returns a dict: {"name": str, "arguments": dict} or None.
        """
        # quick check for the presence of "name" and "arguments"
        if '"name"' not in text:
            return None

        # find index of name key for 'search'
        m = re.search(r'"name"\s*:\s*"search"', text)
        if not m:
            return None

        idx = m.start()

        # expand left to nearest '{'
        start = text.rfind('{', 0, idx)
        if start == -1:
            return None

        # naive brace matching to find JSON object boundary
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            return None

        candidate = text[start:end]
        try:
            obj = json.loads(candidate)
        except Exception:
            # sometimes arguments are nested as a string (function_call style), try to recover
            # try parsing looser: find function_call object
            m2 = re.search(r'"function_call"\s*:\s*(\{.*\})', text, re.DOTALL)
            if m2:
                try:
                    fc = json.loads(m2.group(1))
                    name = fc.get('name')
                    args = fc.get('arguments')
                    # arguments may be a JSON-encoded string
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    return {"name": name, "arguments": args}
                except Exception:
                    return None
            return None

        # normalize shape
        if 'name' in obj and 'arguments' in obj:
            args = obj['arguments']
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return {"name": obj['name'], "arguments": args}

        # sometimes outer wrapper
        if 'function_call' in obj and isinstance(obj['function_call'], dict):
            fc = obj['function_call']
            args = fc.get('arguments')
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return {"name": fc.get('name'), "arguments": args}

        return None

    def _extract_answer(self, text: str) -> Optional[str]:
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    def run(self, question: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        question = question.strip()
        if question and question[-1] != '?':
            question += '?'

        prompt = (
            "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. "
            "You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>. "
            f"Question: {question}\n"
        )

        round_idx = 0
        while round_idx < self.max_rounds:
            round_idx += 1
            # call LLM, provide the search tool schema so model may perform tool calls
            raw = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature, functions=[SEARCH_TOOL_SCHEMA])
            output_text = raw or ""

            # if model emits a tool/function call, handle it
            tool_call = self._extract_tool_call(output_text)
            if tool_call and tool_call.get("name") == "search":
                args = tool_call.get("arguments") or {}
                try:
                    search_results = call_search_with_args(self.searcher, args)
                except Exception as e:
                    # append the error and continue so LLM can react
                    search_results = f"[search error: {e}]"

                append_text = self.curr_search_template.format(output_text=output_text, search_results=search_results)
                prompt += append_text
                continue

            # if there's a final answer, return it
            answer = self._extract_answer(output_text)
            if answer:
                return answer

            # if no search and no answer, but model produced text, append and continue
            prompt += "\n\n" + output_text + "\n\n"

        # if we exhausted rounds, try to extract answer from prompt or last output
        final = self._extract_answer(prompt)
        if final:
            return final
        # fallback: return last generated text
        return output_text.strip()
