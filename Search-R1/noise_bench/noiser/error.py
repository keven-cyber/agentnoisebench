from __future__ import annotations
from typing import Optional, Tuple
import random

from noise_bench.noiser.base import BaseNoiser, NoiserContext
from noise_bench.config import NoiseCommonConfig, ErrorConfig
from noise_bench.api import LLMClient
from noise_bench.system_prompt import zh_add_error, en_add_error

# @dataclass
# class TrailState:
#     used_times: int = 0
#     last_noised_step: Optional[int] = None

class ErrorNoiser(BaseNoiser):
    def __init__(self, common_cfg: NoiseCommonConfig, type_cfg: ErrorConfig):
        super().__init__(common_cfg)
        self.type_cfg = type_cfg

        self.llm = LLMClient(
            model=common_cfg.api.model,
            api_key=common_cfg.api.api_key,
            base_url=common_cfg.api.base_url,
        )

        self._rng = random.Random(self.common_cfg.seed)

        self.system_prompt = self._get_system_prompt()
        # self._state_map: Dict[int, TrailState] = {}

    def _get_system_prompt(self) -> str:
        if self.type_cfg.language == "zh":
            return zh_add_error
        if self.type_cfg.language == "en":
            return en_add_error
        raise ValueError(f"Unsupported language for ErrorNoiser: {self.type_cfg.language}")

    # def _ensure_state(self, trail_id: int) -> TrailState:
    #     st = self._state_map.get(trail_id)
    #     if st is None:
    #         st = TrailState()
    #         self._state_map[trail_id] = st
    #     return st

    def maybe_add_noise(self, content: str, obs: str, ctx: NoiserContext) -> Tuple[bool, str]:
        if not self._should_apply(ctx, obs):
            return (False, obs)

        noisy_text = self._add_noise(content, obs, ctx)
        if not noisy_text:
            return (False, obs)

        # Count a successful application.
        # st = self._ensure_state(int(ctx.trail_id))
        # st.used_times += 1
        # st.last_noised_step = ctx.step_idx
        ctx.added_noise_steps+=1
        ctx.last_noised_step=ctx.step_idx
        return (True, noisy_text)

    def _should_apply(self, ctx: NoiserContext, obs: str) -> bool:
        if not self.common_cfg.enable:
            return False
        if not ctx.is_search:
            return False

        # apply_always branch.
        if self.type_cfg.apply_always:
            # st = self._ensure_state(int(ctx.trail_id))  # original verl-adapted path; requires st
            return True

        # Not apply_always: gated by start step, quota, cooldown and probability.
        # First check whether the current step has reached start_step.
        if (ctx.step_idx + 1) < self.type_cfg.start_step:  # +1 because step_idx is 0-based
            return False

        # Reaching this point means start_step has been satisfied.

        # st = self._ensure_state(int(ctx.trail_id))  # original verl-adapted path; requires st

        # No cap on the number of applications, but interval and probability still apply.
        if self.type_cfg.max_times_per_trail == -1:
            # if st.last_noised_step is not None and (ctx.step_idx - st.last_noised_step) <= self.type_cfg.interval_steps:  # original verl-adapted path; requires st
            if ctx.last_noised_step is not None and (ctx.step_idx - ctx.last_noised_step) <= self.type_cfg.interval_steps:
                return False

            return self._rng.random() <= self.type_cfg.prob_per_step

        # self.type_cfg.max_times_per_trail != -1, i.e. a bounded number of applications.
        # if st.used_times >= self.type_cfg.max_times_per_trail:
        if ctx.used_times >= self.type_cfg.max_times_per_trail:
            return False
        # if st.last_noised_step is not None and (ctx.step_idx - st.last_noised_step) <= self.type_cfg.interval_steps:
        if ctx.last_noised_step is not None and (ctx.step_idx - ctx.last_noised_step) <= self.type_cfg.interval_steps:
            return False

        # Simplest possible probability: one random.random() draw per step.
        return self._rng.random() <= self.type_cfg.prob_per_step


    def _add_noise(self, content: str, obs: str, ctx: NoiserContext) -> Optional[str]:
        user_prompt = self.system_prompt.format(content=content, obs=obs)
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        text = self.llm.chat(
            messages=messages,
            temperature=self.common_cfg.temperature,
        )

        return text
