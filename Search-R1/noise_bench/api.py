
from __future__ import annotations
import time
from typing import Optional, Dict, List

from openai import OpenAI


class LLMClient:
    """
    轻量 OpenAI 兼容客户端封装：
    - 支持 base_url（兼容自建/第三方 OpenAI-compatible 服务）
    - 从指定环境变量读取 API Key
    - 简单重试 + 超时
    - 返回单段文本（不做流式/函数调用）
    """
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_retries: int = 3,
        backoff_base_s: float = 1.0
    ):

        self.model = model
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s


        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                # OpenAI SDK 1.x Chat Completions
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                text = resp.choices[0].message.content
                return text
            except Exception as e:  # pragma: no cover
                last_err = e
                if attempt == self.max_retries - 1:
                    print(f"LLMClient.chat failed after {self.max_retries} attempts.")
                    break
                sleep_s = self.backoff_base_s * (2 ** attempt)
                time.sleep(sleep_s)

        # 最终失败，抛出最后一个异常，交由上层兜底（如 passthrough 原文）
        raise RuntimeError(f"LLMClient.chat failed after retries: {last_err}")
