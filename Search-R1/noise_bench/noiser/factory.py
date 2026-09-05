from typing import Optional
from noise_bench.config import NoiseConfig
from noise_bench.noiser.base import BaseNoiser
from noise_bench.noiser.incomplete import IncompleteNoiser
from noise_bench.noiser.error import ErrorNoiser
from noise_bench.noiser.redundancy import RedundancyNoiser
from noise_bench.noiser.induce import InduceNoiser
from noise_bench.noiser.failure import FailureNoiser
from noise_bench.noiser.mix import MixNoiser

class NoiserFactory:
    @staticmethod
    def get_noiser_instance(cfg: NoiseConfig) -> Optional[BaseNoiser]:
        if not cfg.common.enable:
            print("Note: noise.enable is false.")
            return None
        t = cfg.common.noise_type
        if t == "incomplete":
            return IncompleteNoiser(cfg.common, cfg.incomplete)
        elif t == "error":
            return ErrorNoiser(cfg.common, cfg.error)
        elif t == "redundancy":
            return RedundancyNoiser(cfg.common, cfg.redundancy)
        elif t == "induce":
            return InduceNoiser(cfg.common, cfg.induce)
        elif t == "failure":
            return FailureNoiser(cfg.common, cfg.failure)
        elif t == "mix":
            return MixNoiser(cfg)
        else:
            raise ValueError(f"Unknown noise type: {t}")
