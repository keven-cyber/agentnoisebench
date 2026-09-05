import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI
from agent.searcher import Searcher
from agent.tools import TOOLS
from noise_bench.noiser.base import BaseNoiser, NoiserContext

@dataclass
class AgentConfig:
    model: str
    max_turns: int = 5                    # agent 最多工具调用步数
    max_output_tokens: int = 512          # 每次 LLM 输出上限
    temperature: float = 0.7
    verbose: bool = False                 # 打印调试信息
    # 可以加更多参数（例如 stop sequences 等）


@dataclass
class AgentRunResult:
    question: str
    final_answer: str
    raw_model_answer: str
    steps: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    added_noise_steps: int = 0
    finish_reason: str = ""               # "answer" / "max_turns" / "error"


# =========================
# 一些辅助函数
# =========================

def extract_answer(text: str) -> Optional[str]:
    """
    从模型输出里提取 <answer>...</answer>。
    没有的话返回 None。
    """
    if not text:
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


# =========================
# SearchAgent 主体
# =========================

class SearchAgent:
    """
    一个使用商业 LLM API（OpenAI Responses 兼容格式）的 search agent。

    特点：
    - 工具调用 search_documents -> Searcher.search()
    - 工具返回值先包成 <information>...</information>
    - 然后通过 noiser.maybe_add_noise()（如果提供）加噪，最后丢回 LLM
    - 模型最终必须输出 <answer>...</answer>，用于评估
    """

    def __init__(
        self,
        client: OpenAI,
        config: AgentConfig,
        noiser: BaseNoiser,
        searcher: Optional[Searcher],
    ):
        self.client = client
        self.config = config
        self.searcher = searcher or Searcher()
        self.noiser = noiser

        # system prompt：不再要求 <think>/<search>，但强调 <information> / <answer>
        # self.system_instructions = """Answer the given question. You must conduct reasoning first every time you get new information. After reasoning, if you find you lack some knowledge, you may call a tool named `search_documents` and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. """
        
        self.system_instructions = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you may call a tool named `search_documents` and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. """

    # ---------- 工具调度 ----------

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "search_documents":
            query = arguments.get("query", "")
            topk = int(arguments.get("topk", 3))
            return self.searcher.search(query, topk=topk)
        else:
            raise ValueError(f"未知工具: {tool_name}")

    # ---------- 主推理循环 ----------

    def run(self, question: str) -> AgentRunResult:
        """
        对单个问题运行 search agent。
        """
        question = question.strip()
        
        if question[-1] != '?':
            question += '?'
        
        conversation_history: List[Any] = [
            {"role": "user", "content": question}
        ]
        
        steps = 0
        tool_calls_log: List[Dict[str, Any]] = []
        added_noise_steps = 0 # 记录加噪步骤数
        last_noised_step = None # 记录上一次加噪步骤
        last_text: str = ""
        finish_reason = ""
        
        # 给每一个question创建一个自己的context
        ctx = NoiserContext(
            step_idx=steps,
            is_search=False,
            max_turns=self.config.max_turns,
            trail_id=0, # 就一个question的话，id固定给个0吧
            last_noised_step=last_noised_step,
            added_noise_steps=added_noise_steps,
        )

        response_obj = None  # 防止作用域问题

        while steps < self.config.max_turns:
            if self.config.verbose:
                print(f"\n[Agent] ===== Step {steps} =====")
                print(f"[Agent] 当前 history: {conversation_history}")

            # 调用 LLM
            try:
                response_obj = self.client.responses.create(
                    model=self.config.model,
                    instructions=self.system_instructions,
                    input=conversation_history,
                    tools=TOOLS,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    # tool_choice=""
                )
            except Exception as e:
                finish_reason = f"error: {e}"
                if self.config.verbose:
                    print(f"[Agent] 调用 LLM 出错: {e}")
                break

            if self.config.verbose:
                print(f"[Agent] LLM 原始响应: {response_obj}")

            outputs = response_obj.output or []
            if not outputs:
                if self.config.verbose:
                    print("[Agent] 响应为空，结束。")
                finish_reason = "error_empty_output"
                break

            first = outputs[0]

            # 1）工具调用
            if first.type == "function_call":
                tool_call = first
                tool_name = tool_call.name
                raw_arguments = tool_call.arguments or "{}"
                try:
                    args = json.loads(raw_arguments)
                except Exception:
                    args = {}

                if self.config.verbose:
                    print(f"[Agent] 模型请求调用工具: {tool_name}({args})")

                try:
                    tool_result_plain = self._call_tool(tool_name, args)
                except Exception as e:
                    # 工具失败时的兜底（也可以返回特殊提示，加噪模块可根据 is_search 处理）
                    tool_result_plain = f"工具 {tool_name} 调用失败: {e}"

                if self.config.verbose:
                    print(f"[Tool] 工具返回原始结果: {tool_result_plain}")
                    
                # 包裹 <information>，注意这里是原始结果
                next_ob = f"<information>{tool_result_plain}</information>"
                
                # 噪声注入：只有 search_tools 才 is_search=True
                noisy_next_ob = next_ob
                added_noise = False
                if self.noiser is not None:
                    # 更新 context
                    ctx.step_idx = steps
                    ctx.is_search = True
                    
                    try:
                        added_noise, noisy_next_ob = self.noiser.maybe_add_noise(
                            question, next_ob, ctx
                        )
                    except Exception as e:
                        # 噪声模块异常时不影响主流程
                        if self.config.verbose:
                            print(f"[Agent] noiser 发生异常，使用无噪声结果: {e}")
                        added_noise = False
                        noisy_next_ob = next_ob
                        
                    if self.config.verbose and added_noise:
                        print(f"[Noiser] 加噪前文本: {next_ob}")
                        print(f"[Noiser] 加噪后文本: {noisy_next_ob}")

                    if added_noise:
                        added_noise_steps += 1
                        last_noised_step = steps
                
                else :
                    if self.config.verbose:
                        print("[System] 未提供 noiser，跳过加噪步骤,使用原始结果。")
                        print(f"[System] 原始结果: {next_ob}")
                    
                # 把工具调用 + 工具输出加入对话
                conversation_history.append(tool_call)
                conversation_history.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": noisy_next_ob,
                })

                tool_calls_log.append({
                    "step": steps,
                    "tool_name": tool_name,
                    "arguments": args,
                    "added_noise": added_noise,
                    "raw_output": tool_result_plain,
                    "noisy_output": noisy_next_ob,
                })

                steps += 1
                continue

            # 2）普通 assistant message：看是否已经给出 <answer>
            elif first.type == "message":
                text = response_obj.output_text or ""
                last_text = text
                if self.config.verbose:
                    print(f"[Agent] assistant message: {text}")

                parsed_answer = extract_answer(text)
                if parsed_answer is not None:
                    finish_reason = "answer"
                    return AgentRunResult(
                        question=question,
                        final_answer=parsed_answer,
                        raw_model_answer=text,
                        steps=steps,
                        tool_calls=tool_calls_log,
                        added_noise_steps=added_noise_steps,
                        finish_reason=finish_reason,
                    )
                else:
                    # 没有 <answer>，可能是中间对话，继续
                    conversation_history.append({
                        "role": "assistant",
                        "content": text,
                    })
                    steps += 1
                    continue

            else:
                # 其它类型（理论上不会出现），直接结束
                if self.config.verbose:
                    print(f"[Agent] 未知 output type: {first.type}")
                finish_reason = f"unknown_output_type:{first.type}"
                break

        # 走到这里要么 max_turns，要么 error
        final_text = last_text or (response_obj.output_text if response_obj else "")
        parsed_answer = extract_answer(final_text) or ""
        if not finish_reason:
            finish_reason = "max_turns"

        return AgentRunResult(
            question=question,
            final_answer=parsed_answer,
            raw_model_answer=final_text,
            steps=steps,
            tool_calls=tool_calls_log,
            added_noise_steps=added_noise_steps,
            finish_reason=finish_reason,
        )