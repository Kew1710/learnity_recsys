"""
Diagnostic Computerized Adaptive Test (CAT) for cold start.

Runs a short sequence of 5-10 maximally informative tasks for new students
to rapidly calibrate mastery priors, replacing the pure grade-based prior.

Algorithm:
  1. Start with grade-based prior θ for each KC
  2. Select task that maximizes Fisher information: I(θ) = p(1-p)
     This means targeting tasks where P(correct) ≈ 0.5
  3. After each response, update θ via Bayesian update
  4. Stop when confidence is sufficient or budget exhausted

The module produces a mastery prior update dict that the caller
sends to the Profile service.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


CAT_BUDGET = 8
CAT_MIN_TASKS = 4


@dataclass
class CATState:
    kc_theta: dict[str, float]       # logit-scale ability per KC
    kc_n: dict[str, int]             # attempts per KC
    tasks_used: int = 0

    @classmethod
    def from_mastery(cls, mastery: dict[str, float]) -> CATState:
        theta = {}
        for kc, p in mastery.items():
            p_clamped = max(0.05, min(0.95, p))
            theta[kc] = math.log(p_clamped / (1.0 - p_clamped))
        return cls(kc_theta=theta, kc_n={kc: 0 for kc in mastery})

    def to_mastery(self) -> dict[str, float]:
        return {
            kc: 1.0 / (1.0 + math.exp(-t))
            for kc, t in self.kc_theta.items()
        }

    @property
    def is_complete(self) -> bool:
        if self.tasks_used >= CAT_BUDGET:
            return True
        if self.tasks_used >= CAT_MIN_TASKS:
            tested = [kc for kc, n in self.kc_n.items() if n >= 2]
            return len(tested) >= 3
        return False


def select_diagnostic_kc(state: CATState, available_kcs: list[str]) -> str | None:
    """
    Select the KC that would give most information.
    Prioritizes KCs with fewest attempts, breaking ties by
    choosing KCs where θ is closest to 0 (most uncertain).
    """
    candidates = [kc for kc in available_kcs if kc in state.kc_theta]
    if not candidates:
        return None

    return min(candidates, key=lambda kc: (state.kc_n.get(kc, 0), abs(state.kc_theta.get(kc, 0.0))))


def select_diagnostic_task(
    tasks: list[dict],
    theta: float,
) -> dict | None:
    """
    Select the task that maximizes Fisher information I(θ) = p(1-p).
    This targets tasks where P(correct|θ, difficulty) ≈ 0.5.
    """
    if not tasks:
        return None

    def info_score(task: dict) -> float:
        diff = (task.get("parts") or [{}])[0].get("irt_difficulty", 0.5)
        p = 1.0 / (1.0 + math.exp(-(theta - diff)))
        return p * (1.0 - p)

    return max(tasks, key=info_score)


def update_cat_state(state: CATState, kc_id: str, score: float, irt_difficulty: float) -> None:
    """
    Bayesian update of θ after observing a response.

    Uses a simple EAP-like update:
      θ_new = θ_old + lr * (score - P(correct|θ, difficulty))

    The learning rate decreases as we observe more responses for this KC.
    """
    theta = state.kc_theta.get(kc_id, 0.0)
    n = state.kc_n.get(kc_id, 0)

    p_correct = 1.0 / (1.0 + math.exp(-(theta - irt_difficulty)))
    lr = 1.0 / (1.0 + n)
    state.kc_theta[kc_id] = theta + lr * (score - p_correct)
    state.kc_n[kc_id] = n + 1
    state.tasks_used += 1
