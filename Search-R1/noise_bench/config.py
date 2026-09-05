from dataclasses import dataclass
from typing import Optional, Literal
import os


@dataclass
class SimplifyNoiseConfig:
    api_key: str = os.getenv("OPENAI_API_KEY")
    base_url: str = os.getenv("OPENAI_BASE_URL")

    enable: bool = False
    noise_type: Literal["incomplete", "error", "redundancy", "induce", "failure"] = "failure"
    model: str = "gpt-4o-mini"
    seed: int = 42
    temperature: float = 0.7

    language: Literal["zh", "en"] = "en"
    apply_always: bool = True # if true, noise is applied at every step

    # effective when apply_always=false
    start_step: int = 1
    prob_per_step: float = 1.0
    interval_steps: int = 1
    max_times_per_trail: int = 4


############# the original version follows #############
@dataclass
class NoiseAPIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY")
    base_url: str = os.getenv("OPENAI_BASE_URL")
    model: str = "gpt-4o-mini"

@dataclass
class NoiseCommonConfig:
    api: NoiseAPIConfig = NoiseAPIConfig()

    enable: bool = False
    noise_type: Literal["incomplete", "error", "redundancy", "induce", "failure"] = "failure"
    seed: int = 42
    temperature: float = 0.7


# ===== per-type configuration =====
@dataclass
class IncompleteConfig:
    language: Literal["zh", "en"] = "en"

    apply_always: bool = True # if true, noise is applied at every step

    # effective when apply_always=false
    start_step: int = 1
    prob_per_step: float = 1.0
    interval_steps: int = 1
    max_times_per_trail: int = 4

@dataclass
class FailureConfig:
    apply_always: bool = False          # setting apply_always=true for this noise type is probably too hard for the model

    start_step: int = 1                # effective when apply_always=false. The first step is step 1, so the default of 1 starts noising immediately
    max_times_per_trail: int = -1       # max number of errors per sample. -1 means unbounded, but prob_per_step and interval_steps still apply
    prob_per_step: float = 0.7         # effective when apply_always=false
    interval_steps: int = 0            # minimum number of steps between two noise applications

    # error pool (overridable)
    # error_pool: List[str] = (
    #     # classic HTTP/network related errors
    #     "Error 401 Unauthorized",
    #     "Error 403 Forbidden",
    #     "Error 408 Request Timeout",
    #     "Error 429 Too Many Requests",
    #     "Error 500 Internal Server Error",
    #     "Error 502 Bad Gateway",
    #     "Error 503 Service Unavailable",
    #     "Error 504 Gateway Timeout",
    # )

@dataclass
class ErrorConfig:
    language: Literal["zh", "en"] = "en"

    apply_always: bool = True # if true, noise is applied at every step

    # effective when apply_always=false
    start_step: int = 1
    prob_per_step: float = 1.0
    interval_steps: int = 1
    max_times_per_trail: int = 4


@dataclass
class RedundancyConfig:
    language: Literal["zh", "en"] = "en"

    # trigger control (consistent with the other types)
    apply_always: bool = True
    start_step: int = 1
    prob_per_step: float = 1.0
    interval_steps: int = 1
    max_times_per_trail: int = 4

@dataclass
class InduceConfig:
    language: Literal["zh", "en"] = "en"

    # trigger control (kept consistent with the other types)
    apply_always: bool = True
    start_step: int = 1            # effective when apply_always=false
    prob_per_step: float = 1.0     # effective when apply_always=false
    interval_steps: int = 1
    max_times_per_trail: int = 4

@dataclass
class MixedConfig:
    """
    Placeholder for future support of mixing multiple noise types.
    """
    pass


# ===== aggregate configuration (for passing everything together) =====
@dataclass
class NoiseConfig:
    common: NoiseCommonConfig
    incomplete: Optional[IncompleteConfig] = None
    error: Optional[ErrorConfig] = None
    failure: Optional[FailureConfig] = None
    redundancy: Optional[RedundancyConfig] = None
    induce: Optional[InduceConfig] = None
