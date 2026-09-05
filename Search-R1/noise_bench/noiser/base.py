from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from noise_bench.config import NoiseCommonConfig

@dataclass
class NoiserContext:
    """
    Scalable runtime context.
    """
    step_idx: int                    # 0-based step/episode index in this trail
    is_search: bool                  # whether this step is a search observation
    max_turns: int                   # max_turns from the configuration
    trail_id: Optional[int] = None   # optional: current sample/trajectory id
    added_noise_steps: Optional[int] = None # number of steps that received noise
    last_noised_step: Optional[int] = None  # index of the last noised step

class BaseNoiser:
    """Abstract base class for all Noisers: the glue layer between quota/sampling
    control and the concrete add_noise implementations."""
    def __init__(self, common_cfg:NoiseCommonConfig):
        self.common_cfg = common_cfg          # NoiseCommonConfig

    def maybe_add_noise(self, content: str, obs: str, ctx: NoiserContext) -> Tuple[bool, str]:
        """
        :param content: the original question
        :type content: str
        :param obs: the tool return value
        :type obs: str
        :param ctx: the noise context
        :type ctx: NoiserContext
        :return: the noising result, (noise_applied, noised_text)
        :rtype: Tuple[bool, str]
        """

        raise NotImplementedError
    def _add_noise(self, content: str, obs: str, ctx: NoiserContext) -> Optional[str]:
        raise NotImplementedError
