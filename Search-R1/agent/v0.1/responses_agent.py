import os
import json
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

from .searcher import Searcher


# Tool schema for OpenAI Responses (function-call style)
SEARCH_TOOL = {
    "type": "function",
    "name": "search_documents",
    "description": "Search the local retrieval service and return formatted passages.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string."},
            "topk": {"type": "integer", "description": "Number of results to return.", "default": 3},
        },
        "required": ["query"],
    },
}


# Mapping from tool name to actual Python implementation
def _search_documents_impl(query: str, topk: int = 3) -> str:
    """Call local Searcher and return formatted results."""
    s = Searcher()
    return s.search(query, topk=topk)


TOOL_IMPLS: Dict[str, Any] = {
    "search_documents": _search_documents_impl,
}


SYSTEM_INSTRUCTIONS = (
    "你是一个基于检索增强的问答助手（RAG）。"
    "对每个用户问题，你必须先调用一次 `search_documents` 工具获取上下文，"
    "再基于工具返回内容进行回答；禁止在没有调用工具的情况下直接回答。"
)


def _extract_function_call(resp: dict) -> Optional[dict]:
    """Extract a function call object from the Responses API response.

    Responses API typically contains function call info under
    response.output where an item has type == 'function_call'. We accept
    a few common shapes and return dict with keys: name, arguments (dict), call_id.
    """
    output = resp.get("output") or []
    if not isinstance(output, list):
        return None
    for item in output:
        if getattr(item, "type", None) == "function_call":
            # item is likely a dict-like
            name = item.get("name")
            args_raw = item.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            return {"name": name, "arguments": args, "call_id": item.get("call_id")}
        # fallback older shape
        if isinstance(item, dict) and item.get("type") == "tool_call":
            name = item.get("tool") or item.get("name")
            args = item.get("args") or item.get("arguments") or {}
            return {"name": name, "arguments": args, "call_id": item.get("id")}
    return None


def rag_assistant(
    user_message: str,
    conversation_history: Optional[List[dict]] = None,
    model: str = "gpt-4o",
    client: Optional[OpenAI] = None,
) -> Tuple[str, List[dict]]:
    """RAG assistant using OpenAI Responses API style tools.

    - Calls the model once, checks for a function/tool call.
    - If model requests search_documents, executes the local search and appends
      the tool result to the conversation, then calls model again to produce final answer.

    Returns (answer_text, updated_conversation_history)
    """
    if conversation_history is None:
        conversation_history = []

    # append user message
    conversation_history.append({"role": "user", "content": user_message})

    client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # first call: model may request a tool
    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=conversation_history,
        tools=[SEARCH_TOOL],
    )

    # Try to find a function call
    fc = _extract_function_call(resp)
    if fc and fc.get("name") in TOOL_IMPLS:
        name = fc["name"]
        args = fc.get("arguments") or {}
        # execute tool
        try:
            tool_result = TOOL_IMPLS[name](**args)
        except Exception as e:
            tool_result = f"[tool error: {e}]"

        # append the tool call event and tool output following Responses API pattern
        conversation_history.append({"type": "function_call", "name": name, "arguments": args})
        conversation_history.append({"type": "function_call_output", "call_id": fc.get("call_id"), "output": tool_result})

        # second call: model should synthesize final answer
        second = client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=conversation_history,
            tools=[SEARCH_TOOL],
        )

        # Prefer output_text if present
        answer = getattr(second, "output_text", None) or second.get("output_text") or ""
        # Append assistant reply to history
        conversation_history.append({"role": "assistant", "content": answer})
        return answer, conversation_history

    # If no tool call, just return the model output
    answer = getattr(resp, "output_text", None) or resp.get("output_text") or ""
    conversation_history.append({"role": "assistant", "content": answer})
    return answer, conversation_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single RAG query via Responses-style agent")
    parser.add_argument("--model", type=str, default=os.getenv("AGENT_MODEL", "gpt-4o"))
    parser.add_argument("--question", type=str, help="Question to ask the agent")
    args = parser.parse_args()

    if not args.question:
        print("Please provide --question\n")
        raise SystemExit(1)

    ans, history = rag_assistant(args.question, model=args.model)
    print(json.dumps({"answer": ans, "history_len": len(history)}, ensure_ascii=False, indent=2))
