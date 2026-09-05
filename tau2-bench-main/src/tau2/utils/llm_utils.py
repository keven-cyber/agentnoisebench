# # 将这部分内容完全替换成vitabench中的内容,在其他的文件中只使用了generate和get_cost函数
# import json
# import re
# from typing import Any, Optional

# import litellm
# from litellm import completion, completion_cost
# from litellm.caching.caching import Cache
# from litellm.main import ModelResponse, Usage
# from loguru import logger

# from tau2.config import (
#     DEFAULT_LLM_CACHE_TYPE,
#     DEFAULT_MAX_RETRIES,
#     LLM_CACHE_ENABLED,
#     REDIS_CACHE_TTL,
#     REDIS_CACHE_VERSION,
#     REDIS_HOST,
#     REDIS_PASSWORD,
#     REDIS_PORT,
#     REDIS_PREFIX,
#     USE_LANGFUSE,
# )
# from tau2.data_model.message import (
#     AssistantMessage,
#     Message,
#     SystemMessage,
#     ToolCall,
#     ToolMessage,
#     UserMessage,
# )
# from tau2.environment.tool import Tool

# # litellm._turn_on_debug()

# if USE_LANGFUSE:
#     # set callbacks
#     litellm.success_callback = ["langfuse"]
#     litellm.failure_callback = ["langfuse"]

# litellm.drop_params = True

# if LLM_CACHE_ENABLED:
#     if DEFAULT_LLM_CACHE_TYPE == "redis":
#         logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
#         litellm.cache = Cache(
#             type=DEFAULT_LLM_CACHE_TYPE,
#             host=REDIS_HOST,
#             port=REDIS_PORT,
#             password=REDIS_PASSWORD,
#             namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
#             ttl=REDIS_CACHE_TTL,
#         )
#     elif DEFAULT_LLM_CACHE_TYPE == "local":
#         logger.info("LiteLLM: Using local cache")
#         litellm.cache = Cache(
#             type="local",
#             ttl=REDIS_CACHE_TTL,
#         )
#     else:
#         raise ValueError(
#             f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
#         )
#     litellm.enable_cache()
# else:
#     logger.info("LiteLLM: Cache is disabled")
#     litellm.disable_cache()


# ALLOW_SONNET_THINKING = False

# if not ALLOW_SONNET_THINKING:
#     logger.warning("Sonnet thinking is disabled")

# # 微调模型名称解析函数：提取的基础模型名（如从 "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg"提取 "gpt-4.1-mini-2025-04-14"
# def _parse_ft_model_name(model: str) -> str:
#     """
#     Parse the ft model name from the litellm model name.
#     e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
#     """
#     pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
#     match = re.match(pattern, model)
#     if match:
#         return match.group("model")
#     else:
#         return model

# # 响应成本计算函数，计算单次API调用的成本。
# def get_response_cost(response: ModelResponse) -> float:
#     """
#     Get the cost of the response from the litellm completion.
#     """
#     response.model = _parse_ft_model_name(
#         response.model
#     )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
#     try:
#         cost = completion_cost(completion_response=response)
#     except Exception as e:
#         logger.error(e)
#         return 0.0
#     return cost

# # get_response_usage- Token使用量提取函数
# def get_response_usage(response: ModelResponse) -> Optional[dict]:
#     usage: Optional[Usage] = response.get("usage")
#     if usage is None:
#         return None
#     return {
#         "completion_tokens": usage.completion_tokens,
#         "prompt_tokens": usage.prompt_tokens,
#     }


# def to_tau2_messages(
#     messages: list[dict], ignore_roles: set[str] = set()
# ) -> list[Message]:
#     """
#     Convert a list of messages from a dictionary to a list of Tau2 messages.
#     """
#     tau2_messages = []
#     for message in messages:
#         role = message["role"]
#         if role in ignore_roles:
#             continue
#         if role == "user":
#             tau2_messages.append(UserMessage(**message))
#         elif role == "assistant":
#             tau2_messages.append(AssistantMessage(**message))
#         elif role == "tool":
#             tau2_messages.append(ToolMessage(**message))
#         elif role == "system":
#             tau2_messages.append(SystemMessage(**message))
#         else:
#             raise ValueError(f"Unknown message type: {role}")
#     return tau2_messages


# def to_litellm_messages(messages: list[Message]) -> list[dict]:
#     """
#     Convert a list of Tau2 messages to a list of litellm messages.
#     """
#     litellm_messages = []
#     for message in messages:
#         if isinstance(message, UserMessage):
#             litellm_messages.append({"role": "user", "content": message.content})
#         elif isinstance(message, AssistantMessage):
#             tool_calls = None
#             if message.is_tool_call():
#                 tool_calls = [
#                     {
#                         "id": tc.id,
#                         "name": tc.name,
#                         "function": {
#                             "name": tc.name,
#                             "arguments": json.dumps(tc.arguments),
#                         },
#                         "type": "function",
#                     }
#                     for tc in message.tool_calls
#                 ]
#             litellm_messages.append(
#                 {
#                     "role": "assistant",
#                     "content": message.content,
#                     "tool_calls": tool_calls,
#                 }
#             )
#         elif isinstance(message, ToolMessage):
#             litellm_messages.append(
#                 {
#                     "role": "tool",
#                     "content": message.content,
#                     "tool_call_id": message.id,
#                 }
#             )
#         elif isinstance(message, SystemMessage):
#             litellm_messages.append({"role": "system", "content": message.content})
#     return litellm_messages


# def generate(
#     model: str,
#     messages: list[Message],
#     tools: Optional[list[Tool]] = None,
#     tool_choice: Optional[str] = None,
#     **kwargs: Any,
# ) -> UserMessage | AssistantMessage:
#     """
#     Generate a response from the model.

#     Args:
#         model: The model to use.
#         messages: The messages to send to the model.
#         tools: The tools to use.
#         tool_choice: The tool choice to use.
#         **kwargs: Additional arguments to pass to the model.

#     Returns: A tuple containing the message and the cost.
#     """
#     if kwargs.get("num_retries") is None:
#         kwargs["num_retries"] = DEFAULT_MAX_RETRIES

#     if model.startswith("claude") and not ALLOW_SONNET_THINKING:
#         kwargs["thinking"] = {"type": "disabled"}
#     litellm_messages = to_litellm_messages(messages)
#     tools = [tool.openai_schema for tool in tools] if tools else None
#     if tools and tool_choice is None:
#         tool_choice = "auto"
#     try:
#         response = completion(
#             model=model,
#             messages=litellm_messages,
#             tools=tools,
#             tool_choice=tool_choice,
#             **kwargs,
#         )
#     except Exception as e:
#         logger.error(e)
#         raise e
#     cost = get_response_cost(response)
#     usage = get_response_usage(response)
#     response = response.choices[0]
#     try:
#         finish_reason = response.finish_reason
#         if finish_reason == "length":
#             logger.warning("Output might be incomplete due to token limit!")
#     except Exception as e:
#         logger.error(e)
#         raise e
#     assert response.message.role == "assistant", (
#         "The response should be an assistant message"
#     )
#     content = response.message.content
#     tool_calls = response.message.tool_calls or []
#     tool_calls = [
#         ToolCall(
#             id=tool_call.id,
#             name=tool_call.function.name,
#             arguments=json.loads(tool_call.function.arguments),
#         )
#         for tool_call in tool_calls
#     ]
#     tool_calls = tool_calls or None

#     message = AssistantMessage(
#         role="assistant",
#         content=content,
#         tool_calls=tool_calls,
#         cost=cost,
#         usage=usage,
#         raw_data=response.to_dict(),
#     )
#     return message


# def get_cost(messages: list[Message]) -> tuple[float, float] | None:
#     """
#     Get the cost of the interaction between the agent and the user.
#     Returns None if any message has no cost.
#     """
#     agent_cost = 0
#     user_cost = 0
#     for message in messages:
#         if isinstance(message, ToolMessage):
#             continue
#         if message.cost is not None:
#             if isinstance(message, AssistantMessage):
#                 agent_cost += message.cost
#             elif isinstance(message, UserMessage):
#                 user_cost += message.cost
#         else:
#             logger.warning(f"Message {message.role}: {message.content} has no cost")
#             return None
#     return agent_cost, user_cost


# def get_token_usage(messages: list[Message]) -> dict:
#     """
#     Get the token usage of the interaction between the agent and the user.
#     """
#     usage = {"completion_tokens": 0, "prompt_tokens": 0}
#     for message in messages:
#         if isinstance(message, ToolMessage):
#             continue
#         if message.usage is None:
#             logger.warning(f"Message {message.role}: {message.content} has no usage")
#             continue
#         usage["completion_tokens"] += message.usage["completion_tokens"]
#         usage["prompt_tokens"] += message.usage["prompt_tokens"]
#     return usage



import json
import re
import time
from typing import Any, Optional

from loguru import logger
import requests

from tau2.config import (
    models,
    DEFAULT_MAX_RETRIES,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool


class DictToObject:
    """
    Convert dictionary to object with attribute access
    Usage:
    response_obj = DictToObject(response)
    print(response_obj.choices[0].message.content)  # Instead of response["choices"][0]["message"]["content"]
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert object back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, DictToObject) else item for item in value]
            else:
                result[key] = value
        return result


def get_response_cost(usage, model) -> float:
    num_prompt_token = usage["prompt_tokens"]
    num_completion_token = usage["completion_tokens"]
    prompt_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("prompt_price", 0)
    completion_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("completion_price", 0)
    if prompt_price and completion_price:
        return (prompt_price * num_prompt_token + completion_price * num_completion_token) / 1000000
    else:
        return 0.0


def get_response_usage(response) -> Optional[dict]:
    usage = response.get("usage", {})
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0)
    }


def format_messages(messages: list[Message]) -> list[dict]:
    messages_formatted = []
    for message in messages:
        if isinstance(message, UserMessage):
            messages_formatted.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            messages_formatted.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            messages_formatted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                    "name": message.name,
                }
            )
        elif isinstance(message, SystemMessage):
            messages_formatted.append({"role": "system", "content": message.content})
    return messages_formatted


def to_claude_think_official(messages_formatted: list[dict], messages: list[Message]) -> list[dict]:
    try:
        idx = -2 if messages_formatted[-1]["role"] == "tool" else -1
        content = [
            {
                "type": "text",
                "text": messages[idx].content
            }
        ]
        if messages[idx].raw_data["message"].get("tool_calls", []):
            content.append(
                {
                    "type": "tool_use",
                    "id": messages[idx].raw_data["message"]["tool_calls"][0]['id'],
                    "name": messages[idx].raw_data["message"]["tool_calls"][0]["function"]["name"],
                    "input": messages[idx].raw_data["message"]["tool_calls"][0]["function"]["arguments"]
                }
            )
        reasoning_content = messages[idx].raw_data["message"].get("reasoning_content", None) or messages[idx].raw_data["message"].get("reasoning", None)
        if reasoning_content:
            content.append(
                {
                    "type": "thinking",
                    "thinking": reasoning_content
                }
            )

        messages_formatted[idx]["content"] = content
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)

    return messages_formatted


def kwargs_adapter(data: dict, enable_think: False, messages: list) -> dict:
    if "claude" in data["model"]:
        if not enable_think:
            data["thinking"] = {"type": "disabled"}
        else:
            data["messages"] = to_claude_think_official(data["messages"], messages)
    else:
        if not enable_think:
            if data.get("model", "") == "gpt-5":
                data["reasoning_effort"] = "minimal"
            elif "reasoning_effort" in data:
                data.pop("reasoning_effort")
    return data


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    enable_think: bool = False,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        enable_think: Whether to enable think mode for the agent.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    try:
        if kwargs.get("num_retries") is None:
            kwargs["num_retries"] = DEFAULT_MAX_RETRIES
        messages_formatted = format_messages(messages)
        tools = [tool.openai_schema for tool in tools] if tools else None
        if tools and tool_choice is None:
            tool_choice = "auto"
        try:
            data = {
                "model": model,
                "messages": messages_formatted,
                "stream": False,
                "temperature": kwargs.get("temperature"),
                "tools": tools,
                "tool_choice": tool_choice,
            }
            data.update(models[model])
            data = kwargs_adapter(data, enable_think, messages)
            headers = models[model]["headers"]

            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries + 1):
                try:
                    response = requests.post(data["base_url"], json=data, headers=headers, timeout=(10, 600))
                    if response.status_code != 500:
                        response = response.json()
                        break

                    if attempt < max_retries:
                        logger.warning(f"API returned 500 error, attempt {attempt + 1} retry, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        response.raise_for_status()

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries:
                        logger.warning(f"Request exception, attempt {attempt + 1} retry, retrying in {retry_delay} seconds... Error: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise e
        except Exception as e:
            logger.error(e)
            raise e
        usage = get_response_usage(response)
        cost = get_response_cost(usage, model)
        try:
            response = response['choices'][0]
        except Exception:
            print(f"bad response: {response}")
        assert response['message']['role'] == "assistant", (
            "The response should be an assistant message"
        )
        # print("\033[31m" + f"response: {response}" + "\033[0m")
        read1 = response['message'].get('reasoning_content')
        read2 = response['message'].get('content') 
        if read2:  # 简单判断
            content = read2
        else:
            content = read1
        tool_calls = response['message'].get('tool_calls') or []
        tool_calls = [
            ToolCall(
                id=tool_call.get('id'),
                name=tool_call.get('function', {}).get('name'),
                arguments=json.loads(tool_call.get('function', {}).get('arguments')) if tool_call.get('function', {}).get('arguments') else {},
            )
            for tool_call in tool_calls
        ]
        tool_calls = tool_calls or None
        message = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=cost,
            usage=usage,
            raw_data=response,
        )
        return message
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(e)


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage