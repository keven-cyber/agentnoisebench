import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI
from agent.searcher import Searcher
from agent.tools import TOOLS  # Responses-style tools; converted to Chat style below
from noise_bench.noiser.base import BaseNoiser, NoiserContext


@dataclass
class AgentConfig:
    model: str
    max_turns: int = 5                    # maximum number of tool-calling steps for the agent
    max_output_tokens: int = 512          # output cap per LLM call
    temperature: float = 0.7
    verbose: bool = False                 # print debug information
    # Added: thinking-related options
    enable_thinking: Optional[bool] = False
    reasoning_effort: Optional[str] = "high"    # "low" | "medium" | "high" | None

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
# Helper functions
# =========================

def extract_answer(text: str) -> Optional[str]:
    """
    Extracts <answer>...</answer> from the model output.
    Returns None if absent.
    """
    if not text:
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


# =========================
# SearchAgent core (uses Chat Completions)
# =========================

class SearchAgent:
    """
    A search agent driven by a commercial LLM API (OpenAI Chat Completions
    compatible format).

    Characteristics:
    - tool call search_documents -> Searcher.search()
    - the tool return value is first wrapped in <information>...</information>
    - noise is then injected via noiser.maybe_add_noise() (when provided) before
      the result is handed back to the LLM
    - the model must finally emit <answer>...</answer>, which is used for evaluation
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

        self.chat_tools = TOOLS


        self.system_instructions = """Answer the given question. You must conduct reasoning first every time you get new information. After reasoning, if you find you lack some knowledge, you may call a tool named `search_documents` and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. """

    # ---------- tool dispatch ----------

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "search_documents":
            query = arguments.get("query", "")
            topk = int(arguments.get("topk", 3))
            return self.searcher.search(query, topk=topk)
        else:
            raise ValueError(f"未知工具: {tool_name}")

    def _build_llm_call_kwargs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Builds the chat.completions.create parameters from the model name and
        AgentConfig:
        - gpt family: uses the top-level reasoning_effort, no extra_body
        - qwen / glm / claude / deepseek: pass the thinking config via extra_body
        - gemini-2.5: configured through reasoning_effort (e.g. 'none' disables thinking)
        """
        model_name = self.config.model

        # Base parameters (shared by all models)
        kwargs: Dict[str, Any] = dict(
            model=self.config.model,
            messages=messages,
            tools=self.chat_tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            tool_choice="auto",
        )

        extra_body: Dict[str, Any] = {}

        # ===== gpt family =====
        if "gpt" in model_name or "o4" in model_name:
            # Official gpt family: reasoning_effort is a top-level parameter, extra_body is unnecessary.
            # If the user supplied reasoning_effort explicitly, honour it.
            if self.config.reasoning_effort:
                kwargs["reasoning_effort"] = self.config.reasoning_effort


        # ===== qwen family =====
        # https://www.alibabacloud.com/help/zh/model-studio/deep-thinking
        elif "qwen" in model_name:
            # Requirement: extra_body = {"enable_thinking": True}, and reasoning_effort is not used
            if self.config.enable_thinking:
                extra_body["enable_thinking"] = True
            else:
                extra_body["enable_thinking"] = False

        # ===== glm family =====
        # https://docs.z.ai/guides/develop/openai/python#thinking-mode
        elif "glm" in model_name:
            # Requirement: extra_body = {"thinking": {"type": "enabled"}}
            if self.config.enable_thinking:
                extra_body["thinking"] = {"type": "enabled"}
            else:
                extra_body["thinking"] = {"type": "disabled"}

        # ===== claude family =====
        # https://platform.claude.com/docs/en/api/openai-sdk#extended-thinking-support
        elif "claude" in model_name:
            # Requirement: extra_body = {"thinking": {"type": "enabled"}}
            if self.config.enable_thinking:
                extra_body["thinking"] = {"type": "enabled"}
            else:
                extra_body["thinking"] = {"type": "disabled"}

        # ===== deepseek family =====
        # https://api-docs.deepseek.com/guides/thinking_mode
        elif "deepseek" in model_name:
            # Requirement: extra_body = {"thinking": {"type": "enabled"}}
            if self.config.enable_thinking:
                extra_body["thinking"] = {"type": "enabled"}
            else:
                extra_body["thinking"] = {"type": "disabled"}

        # ===== Gemini 2.5 family =====
        # (https://ai.google.dev/gemini-api/docs/openai?_gl=1*107gsy6*_up*MQ..*_ga*MTY5NTUyMzAyLjE3NjQ4MzY0NjM.*_ga_P1DBVKWT6V*czE3NjQ4MzY0NjMkbzEkZzAkdDE3NjQ4MzY0NjMkajYwJGwwJGgzNjQ5ODUwMA..#thinking)
        elif "gemini" in model_name:
            if self.config.enable_thinking:

                kwargs["reasoning_effort"] = "high"
            else:
                # Explicitly disable thinking
                 kwargs["reasoning_effort"] = "none"

        # ===== doubao family =====
        # https://www.volcengine.com/docs/82379/1330626?lang=zh
        elif "doubao" in model_name:
            if self.config.enable_thinking:
                extra_body["thinking"] = {"type": "enabled"}
            else:
                extra_body["thinking"] = {"type": "disabled"}


        # Only attach extra_body when there is something to send; otherwise omit the parameter.
        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs

    # ---------- main reasoning loop (chat.completions) ----------

    def run(self, question: str) -> AgentRunResult:
        """
        Runs the search agent on a single question.
        """
        question = question.strip()
        if question and question[-1] != "?":
            question += "?"

        # messages structure for Chat Completions
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_instructions},
            {"role": "user", "content": question},
        ]

        steps = 0
        tool_calls_log: List[Dict[str, Any]] = []
        added_noise_steps = 0
        last_noised_step = None
        last_text: str = ""
        finish_reason = ""

        # Create a dedicated context for every question
        ctx = NoiserContext(
            step_idx=steps,
            is_search=False,
            max_turns=self.config.max_turns,
            trail_id=0,  # with a single question, a fixed id of 0 is enough
            last_noised_step=last_noised_step,
            added_noise_steps=added_noise_steps,
        )

        completion_obj = None  # guard against scoping issues

        while steps < self.config.max_turns:

            if self.config.verbose:
                print(f"\n[Agent] ===== Step {steps} =====")
                print(f"[Agent] current messages: {messages}")

            # Call Chat Completions
            try:
                llm_kwargs = self._build_llm_call_kwargs(messages)
                completion_obj = self.client.chat.completions.create(**llm_kwargs)
            except Exception as e:
                finish_reason = f"error: {e}"
                if self.config.verbose:
                    print(f"[Agent] LLM call failed: {e}")
                break

            if self.config.verbose:
                print(f"[Agent] raw LLM response: {completion_obj}")

            if not completion_obj.choices:
                if self.config.verbose:
                    print("[Agent] empty response, stopping.")
                finish_reason = "error_empty_output"
                break

            choice = completion_obj.choices[0]
            message = choice.message

            # 1) A tool call was requested
            if message.tool_calls:
                if self.config.verbose:
                    print(f"[Agent] model requested tool calls: {message.tool_calls}")

                # First append the assistant's tool_calls message to the history
                assistant_msg = {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [],
                }
                for tc in message.tool_calls:
                    assistant_msg["tool_calls"].append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
                messages.append(assistant_msg)

                # Execute each tool_call in order
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    raw_arguments = tc.function.arguments or "{}"
                    try:
                        args = json.loads(raw_arguments)
                    except Exception:
                        args = {}

                    if self.config.verbose:
                        print(f"[Agent] tool call: {tool_name}({args})")

                    try:
                        tool_result_plain = self._call_tool(tool_name, args)
                    except Exception as e:
                        tool_result_plain = f"工具 {tool_name} 调用失败: {e}"

                    if self.config.verbose:
                        print(f"[Tool] raw tool output: {tool_result_plain}")

                    # Wrap in <information>
                    next_ob = f"<information>{tool_result_plain}</information>"

                    # Noise injection
                    noisy_next_ob = next_ob
                    added_noise = False
                    if self.noiser is not None:
                        ctx.step_idx = steps
                        ctx.is_search = True
                        try:
                            added_noise, noisy_next_ob = self.noiser.maybe_add_noise(
                                question, next_ob, ctx
                            )
                        except Exception as e:
                            if self.config.verbose:
                                print(f"[Agent] noiser raised, falling back to the noise-free result: {e}")
                            added_noise = False
                            noisy_next_ob = next_ob

                        if self.config.verbose and added_noise:
                            print(f"[Noiser] text before noise: {next_ob}")
                            print(f"[Noiser] text after noise: {noisy_next_ob}")

                        if added_noise:
                            added_noise_steps += 1
                            last_noised_step = steps
                            # the counters inside ctx update themselves

                    else:
                        if self.config.verbose:
                            print("[System] no noiser provided, skipping the noise step and using the raw result.")
                            print(f"[System] raw result: {next_ob}")

                    # Append the tool output to messages (the tool message of Chat Completions)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": noisy_next_ob,
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

            # 2) Plain assistant text message: check whether <answer> has been produced
            else:
                # message.content is normally a str; if it is a list (multi-part), join it
                text = message.content or ""
                if isinstance(text, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in text
                    )
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
                    # No <answer> yet; this may be intermediate reasoning, so keep feeding the model
                    messages.append({
                        "role": "assistant",
                        "content": text,
                    })
                    steps += 1
                    continue

        # Reaching here means either max_turns or an error
        final_text = last_text
        if completion_obj and not final_text and completion_obj.choices:
            msg = completion_obj.choices[0].message
            final_text = msg.content or ""

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
