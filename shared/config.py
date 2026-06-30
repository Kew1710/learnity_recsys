"""Centralized configuration for all ML/system constants.

Values can be overridden via environment variables with the LEARNITY_ prefix.
Example: LEARNITY_SMOOTH_LR=0.20 overrides SMOOTH_LR.
"""

import os
from dataclasses import dataclass, fields


def _env(name: str, default):
    val = os.getenv(f"LEARNITY_{name}")
    if val is None:
        return default
    return type(default)(val)


@dataclass(frozen=True)
class RetrievalConfig:
    CLUSTER_EXPLORE_RATE: float = 0.20
    CLUSTER_EXPLORE_THRESHOLD: int = 3
    EPSILON_GREEDY_RATE: float = 0.05
    KC_COOLDOWN_WINDOW: int = 6
    KC_COOLDOWN_MAX: int = 3
    PHASE1_TASK_THRESHOLD: int = 15
    CONTEXT_DIM: int = 13
    ALPHA: float = 0.5          # LinUCB control group (A/B baseline)
    V_SQUARED: float = 0.25     # Thompson Sampling posterior variance scale
    TARGET_ZPD_ACCURACY: float = 0.65
    SUMMARY_WINDOW: int = 20


@dataclass(frozen=True)
class BKTConfig:
    SMOOTH_LR: float = 0.15
    SMOOTH_TRANSIT: float = 0.02
    SURPRISE_K: float = 1.0
    HALF_LIFE_DAYS: float = 30.0
    PERFORMANCE_DECAY_THRESHOLD: int = 3
    PERFORMANCE_DECAY_FACTOR: float = 0.75
    CONFIDENCE_ATTEMPTS_SCALE: int = 10


@dataclass(frozen=True)
class ZPDConfig:
    MASTERY_THRESHOLD: float = 0.7
    MASTERY_CEILING: float = 0.95
    STRONG_PREREQ: float = 0.5


@dataclass(frozen=True)
class GatewayConfig:
    REWARD_BETA: float = 0.3
    REWARD_GAMMA: float = 0.1


@dataclass(frozen=True)
class ClusteringConfig:
    N_CLUSTERS: int = 15


def _load(cls):
    kwargs = {}
    for f in fields(cls):
        kwargs[f.name] = _env(f.name, f.default)
    return cls(**kwargs)


retrieval = _load(RetrievalConfig)
bkt = _load(BKTConfig)
zpd = _load(ZPDConfig)
gateway = _load(GatewayConfig)
clustering = _load(ClusteringConfig)
