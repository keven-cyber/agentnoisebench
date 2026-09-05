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
        # 这里原本是访问你的检索服务的代码，先用示例返回占位
        # payload = {"queries": [query], "topk": topk, "return_scores": True}
        # resp = requests.post(self.retrieve_url, json=payload, timeout=15)
        # resp.raise_for_status()
        # results = resp.json().get("result")
        # if not results:
        #     return ""
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


# ========= 自定义工具（function tool）定义 =========

# 真正调用本地检索的实现
def search_documents(query: str, topk: int = 3) -> str:
    """
    调用本地检索服务，并返回格式化后的文档字符串。
    """
    return searcher.search(query, topk=topk)


# Chat Completions 工具定义格式
TOOLS = [
    {
        "type": "function",
        "function": {
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
        },
    }
]

# 工具名到实际 Python 函数的映射
TOOL_IMPLS = {
    "search_documents": search_documents,
}


# ========= 一个简单的 RAG 助手封装（单轮 or 多轮都可以用） =========

def rag_assistant(user_message: str, conversation_history: list):
    """
    使用 Chat Completions + function tool 调用本地检索服务。

    - conversation_history: 标准 Chat Completions messages（不包含 system），你可以一直传下去。
      例如：
      [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "assistant", "tool_calls": [...]},
        {"role": "tool", "tool_call_id": "...", "name": "search_documents", "content": "..."},
      ]
    - 函数返回 (answer, new_history)
    """
    if conversation_history is None:
        conversation_history = []

    # 1. 把当前用户消息加入对话历史（不含 system）
    conversation_history.append({
        "role": "user",
        "content": user_message,
    })

    print(f"[DEBUG] 1. 当前 conversation_history: {conversation_history}")

    # 构造发给 API 的 messages，在最前面加上 system 指令
    api_messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        *conversation_history,
    ]

    # 2. 第一次调用：让模型决定是否调用工具
    try:
        first_response = client.chat.completions.create(
            # model="deepseek-r1-0528",
            model="gpt-4o",
            messages=api_messages,
            tools=TOOLS,
            tool_choice="auto",  # 让模型自己决定是否调用工具
        )
    except Exception as e:
        raise RuntimeError(f"调用 Chat Completions 失败: {e}")

    print(f"[DEBUG] 模型首次响应: {first_response}")

    first_msg = first_response.choices[0].message
    
    conversation_history.append(first_msg)
    
    tool_calls = getattr(first_msg, "tool_calls", None)

    # 3. 如果模型要调用工具
    if tool_calls:
        # 这里只处理单个工具调用（一般 RAG 场景够用）
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments or "{}"
        args = json.loads(raw_arguments)

        print(f"[DEBUG] 模型请求调用工具: {tool_name}({args})")

        # 实际执行本地工具（检索）
        if tool_name in TOOL_IMPLS:
            tool_result = TOOL_IMPLS[tool_name](**args)
            print(f"[DEBUG] 工具返回结果: {tool_result}")
        else:
            tool_result = f"未知工具: {tool_name}"

        # # 把“assistant 的工具调用消息”加入对话历史
        # conversation_history.append({
        #     "role": "assistant",
        #     "tool_calls": [
        #         {
        #             "id": tool_call.id,
        #             "type": "function",
        #             "function": {
        #                 "name": tool_name,
        #                 "arguments": raw_arguments,
        #             },
        #         }
        #     ],
        # })
        

        print(f"[DEBUG] 2. 当前 conversation_history: {conversation_history}")

        # 再把“tool 的返回结果”消息加入对话历史
        conversation_history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": tool_result,  # 可以是字符串或 json.dumps 出来的字符串
        })

        print(f"[DEBUG] 3. 当前 conversation_history: {conversation_history}")

        # 4. 第二次调用：让模型基于工具结果生成最终回答
        api_messages_second = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            *conversation_history,
        ]

        second_response = client.chat.completions.create(
            # model="deepseek-r1-0528",
            model="gpt-4o",
            messages=api_messages_second,
        )

        print(f"[DEBUG] 模型二次响应: {second_response}")

        answer = second_response.choices[0].message.content

        # 把最终回答加入历史
        conversation_history.append({
            "role": "assistant",
            "content": answer,
        })

        return answer, conversation_history

    # 5. 如果模型认为不需要调用工具（按你的 system 指令一般不会出现）
    else:
        answer = first_msg.content
        conversation_history.append({
            "role": "assistant",
            "content": answer,
        })
        return answer, conversation_history


# ========= 一个极简命令行交互 Demo =========

if __name__ == "__main__":
    history: list = []

    user_input = "Who wrote Sapiens?"

    try:
        answer, history = rag_assistant(user_input, history)
        print("\n助手 answer：", answer)

        print(f"\n[INFO] 当前对话历史记录：{history}")
    except Exception as e:
        print(f"\n[ERROR] 调用失败: {e}")
