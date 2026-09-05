# Planned extension towards composite (non-single-type) noise.

from __future__ import annotations
from typing import Optional, Tuple

from noise_bench.noiser.base import NoiserContext

class MixNoiser:
    def __init__(self, noisers, weights=None):
        pass

    def add_noise(self, text) -> Optional[str]:
        pass

    def maybe_add_noise(self, content: str, obs: str, ctx: NoiserContext) -> Tuple[bool, str]:
        pass
