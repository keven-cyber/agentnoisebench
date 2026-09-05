import json
from typing import List

from openai import OpenAI


# ========= 本地检索封装（用你给的 Searcher） =========

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
        # payload = {"queries": [query], "topk": topk, "return_scores": True}
        # resp = requests.post(self.retrieve_url, json=payload, timeout=15)
        # resp.raise_for_status()
        # results = resp.json().get("result")
        # if not results:
        #     return ""
        # # return formatted string for the first query
        # return self._passages2string(results[0])
        
        return "这是检索到的文档内容示例1。\n这是检索到的文档内容示例2。\n这是检索到的文档内容示例3。"


searcher = Searcher() 


# ========= OpenAI 客户端 =========

# 这里显式从环境变量里读，保证能兼容各种代理/反向代理服务
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_BASE_URL"),  # 例如 https://api.openai.com/v1 或你的代理地址
# )

client = OpenAI(
    api_key="<api key>",
    base_url="https://www.blueshirtmap.com/v1",  # 例如 https://api.openai.com/v1 或你的代理地址
)

SYSTEM_INSTRUCTIONS = (
    "你是一个基于检索增强的问答助手（RAG）。"
    "对每个用户问题，你必须先调用一次 `search_documents` 工具获取上下文，"
    "再基于工具返回内容进行回答；禁止在没有调用工具的情况下直接回答。"
)

# SYSTEM_INSTRUCTIONS = "You are a helpful assistant."


# ========= 自定义工具（function tool）定义 =========

# 这个 Python 函数是真正调用本地检索的实现
def search_documents(query: str, topk: int = 3) -> str:
    """
    调用本地检索服务，并返回格式化后的文档字符串。
    """
    return searcher.search(query, topk=topk)


# 按照 Responses API 的工具（function）定义规范：
# - type: "function"
# - name / description / parameters / strict
# 注意这里的结构是直接在工具对象上写 name/parameters，
# 而不是老版 Chat Completions 那种嵌套在 "function": {...} 里面的格式。:contentReference[oaicite:1]{index=1}
TOOLS = [
    {
        "type": "function",
        "name": "search_documents",
        "description": "在线检索服务，根据用户问题检索相关文档片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户问题或查询语句，会直接用于在线检索。",
                },
                "topk": {
                    "type": "integer",
                    "description": "要返回的文档数量（1-10 条）。",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

# 工具名到实际 Python 函数的映射，方便之后调度
TOOL_IMPLS = {
    "search_documents": search_documents,
}


# ========= 一个简单的 RAG 助手封装（单轮 or 多轮都可以用） =========

def rag_assistant(user_message: str, conversation_history: list):
    """
    使用 Responses API + function tool 调用本地检索服务。

    - conversation_history 用来支持多轮对话，你可以把它一直传下去。
    - 函数返回 (answer, new_history)
    """
    if conversation_history is None:
        conversation_history = []

    # 把当前用户消息加入对话
    conversation_history.append({
        "role": "user",
        "content": user_message,
    })
    
    print(f"[DEBUG] 1. 当前conversation_history: {conversation_history}")  

    # 第一次调用：让模型决定要不要调用工具
    try:
        first_response = client.responses.create(
            model="deepseek-r1-0528",
            instructions=SYSTEM_INSTRUCTIONS,
            input=conversation_history,
            tools=TOOLS,
        )
    except Exception as e:
        raise RuntimeError(f"调用 Responses API 失败: {e}")
    
    print(f"[DEBUG] 模型首次响应: {first_response}")

    # Responses API 中，工具调用会出现在 response.output 列表里，
    # type == "function_call" 表示模型想调用自定义函数。:contentReference[oaicite:2]{index=2}
    if (
        first_response.output
        and isinstance(first_response.output, list)
        and first_response.output[0].type == "function_call"
    ):
        tool_call = first_response.output[0]

        tool_name = tool_call.name # 模型请求调用的工具名
        raw_arguments = tool_call.arguments or "{}" # 工具调用参数（JSON 字符串）
        args = json.loads(raw_arguments) # 解析成字典

        print(f"[DEBUG] 模型请求调用工具: {tool_name}({args})")

        # 实际执行本地工具（这里就是调用你的检索服务）
        if tool_name in TOOL_IMPLS:
            tool_result = TOOL_IMPLS[tool_name](**args)
            
            print(f"[DEBUG] 工具返回结果: {tool_result}")
            
        else:
            tool_result = f"未知工具: {tool_name}"

        # 把工具调用事件 + 工具返回结果，按 Responses API 规范追加回 input：
        # 1）先把 function_call 本身加回对话
        conversation_history.append(tool_call)
        
        print(f"[DEBUG] 2. 当前conversation_history: {conversation_history}")

        # 2）再加一条 type 为 function_call_output 的“工具结果”消息，
        #    里面带上 call_id 和 output 字段。:contentReference[oaicite:3]{index=3}
        conversation_history.append({
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            # 这里直接把检索结果字符串返回给模型；
            # 如果你想返回结构化数据，可以用 json.dumps({...})
            "output": tool_result,
        })
        
        print(f"[DEBUG] 3. 当前conversation_history: {conversation_history}")

        # 第二次调用：让模型基于工具结果生成最终回答
        second_response = client.responses.create(
            model="deepseek-r1-0528",
            instructions=SYSTEM_INSTRUCTIONS,
            input=conversation_history,
            tools=TOOLS,
        )

        print(f"[DEBUG] 模型二次响应: {second_response}")
        answer = second_response.output_text
        
        # 把模型最终回答也加入对话历史
        conversation_history.append({
            "role": "assistant",
            "content": answer,
        })
        
        return answer, conversation_history

    else:
        # 模型认为不需要调用工具，直接回答
        answer = first_response.output_text
        return answer, conversation_history


# ========= 一个极简命令行交互 Demo =========

if __name__ == "__main__":
    history: list = []

    user_input = "Who wrote Sapiens?"
        
    try:
        answer, history = rag_assistant(user_input, history)
        print("\n助手answer：", answer)
        
        print(f"\n[INFO] 当前对话历史记录：{history}")
    except Exception as e:
        print(f"\n[ERROR] 调用失败: {e}")
