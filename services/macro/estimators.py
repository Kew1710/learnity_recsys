"""Rule-based macro estimators built on top of MacroStudentProfile."""

from __future__ import annotations

import random

from .student_profile import MacroStudentProfile
from .tasks_estimator import estimate as estimate_base_tasks


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def estimate_tasks_to_mastery(
    profile: MacroStudentProfile,
    kc_id: str,
    current_mastery: float,
    target_mastery: float,
    weak_prereq_fraction: float,
    confidence: float | None = None,
    cluster_avg: float | None = None,
    n_practiced_in_subject: int = 0,
    n_sims: int = 300,
    rng: random.Random | None = None,
) -> int:
    """Profile-aware adjustment over the base tasks estimator."""
    base = estimate_base_tasks(
        m_current=current_mastery,
        m_target=target_mastery,
        cluster_avg=cluster_avg,
        n_practiced_in_subject=n_practiced_in_subject,
        n_sims=n_sims,
        rng=rng,
    )
    confidence_penalty = 1.0 + max(0.0, 0.6 - (confidence if confidence is not None else profile.mastery_confidence_mean)) * 0.8
    prereq_penalty = 1.0 + 0.5 * weak_prereq_fraction
    uncertainty_penalty = 1.0 + 0.35 * profile.uncertainty_level
    tasks = base * profile.budget_multiplier * confidence_penalty * prereq_penalty * uncertainty_penalty
    return max(1, int(round(tasks)))


def estimate_stall_risk(
    profile: MacroStudentProfile,
    kc_id: str,
    weak_prereq_fraction: float,
    confidence: float | None = None,
    graph_depth: int | None = None,
) -> float:
    """Rule-based probability of stalling on a KC."""
    depth_penalty = min(0.2, 0.03 * max(0, graph_depth or 0))
    confidence_component = 1.0 - (confidence if confidence is not None else profile.mastery_confidence_mean)
    risk = (
        0.45 * profile.stall_risk_baseline
        + 0.25 * weak_prereq_fraction
        + 0.2 * confidence_component
        + depth_penalty
    )
    return round(clamp(risk), 4)


def estimate_regression_risk(
    profile: MacroStudentProfile,
    kc_id: str,
    confidence: float | None = None,
) -> float:
    """Rule-based probability that mastery decays after apparent success."""
    confidence_component = 1.0 - (confidence if confidence is not None else profile.mastery_confidence_mean)
    risk = (
        0.6 * profile.regression_risk_baseline
        + 0.25 * confidence_component
        + 0.15 * profile.uncertainty_level
    )
    return round(clamp(risk), 4)
