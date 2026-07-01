"""Tests for profile-aware macro estimators."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

from services.macro.estimators import (
    estimate_regression_risk,
    estimate_stall_risk,
    estimate_tasks_to_mastery,
)
from services.macro.student_profile import MacroStudentProfile


def make_profile(
    *,
    uncertainty_level: float = 0.2,
    mastery_confidence_mean: float = 0.7,
    weak_prereq_fraction: float = 0.2,
    learning_speed_global: float = 0.05,
    learning_speed_recent: float = 0.05,
    frustration_risk: float = 0.15,
    stall_risk_baseline: float = 0.2,
    regression_risk_baseline: float = 0.2,
    pacing_mode: str = "balanced",
    budget_multiplier: float = 1.0,
    prereq_strictness: float = 0.4,
    test_readiness_bias: float = 0.6,
    step_granularity: float = 1.0,
) -> MacroStudentProfile:
    return MacroStudentProfile(
        student_id=uuid.uuid4(),
        version=1,
        updated_at=datetime.now(timezone.utc),
        target_kc_id="kc_target",
        uncertainty_level=uncertainty_level,
        mastery_confidence_mean=mastery_confidence_mean,
        weak_prereq_fraction=weak_prereq_fraction,
        target_subgraph_mastery_mean=0.5,
        learning_speed_global=learning_speed_global,
        learning_speed_recent=learning_speed_recent,
        tasks_to_gain_01_mastery=2.0,
        recovery_after_error=0.04,
        frustration_risk=frustration_risk,
        stall_risk_baseline=stall_risk_baseline,
        regression_risk_baseline=regression_risk_baseline,
        pacing_mode=pacing_mode,
        budget_multiplier=budget_multiplier,
        prereq_strictness=prereq_strictness,
        test_readiness_bias=test_readiness_bias,
        step_granularity=step_granularity,
    )


def test_tasks_to_mastery_increases_for_risky_profile():
    low_risk = make_profile(uncertainty_level=0.1, budget_multiplier=0.9)
    high_risk = make_profile(uncertainty_level=0.6, budget_multiplier=1.3)

    low = estimate_tasks_to_mastery(
        low_risk,
        kc_id="kc_target",
        current_mastery=0.4,
        target_mastery=0.8,
        weak_prereq_fraction=0.1,
        confidence=0.8,
        n_sims=100,
        rng=random.Random(42),
    )
    high = estimate_tasks_to_mastery(
        high_risk,
        kc_id="kc_target",
        current_mastery=0.4,
        target_mastery=0.8,
        weak_prereq_fraction=0.5,
        confidence=0.2,
        n_sims=100,
        rng=random.Random(42),
    )

    assert high > low


def test_stall_risk_rises_with_weak_prereqs_and_depth():
    profile = make_profile(stall_risk_baseline=0.25, mastery_confidence_mean=0.7)
    shallow = estimate_stall_risk(profile, "kc_x", weak_prereq_fraction=0.1, confidence=0.8, graph_depth=1)
    deep = estimate_stall_risk(profile, "kc_x", weak_prereq_fraction=0.6, confidence=0.2, graph_depth=5)
    assert deep > shallow


def test_regression_risk_rises_when_confidence_is_low():
    profile = make_profile(regression_risk_baseline=0.2, uncertainty_level=0.3)
    low = estimate_regression_risk(profile, "kc_x", confidence=0.9)
    high = estimate_regression_risk(profile, "kc_x", confidence=0.2)
    assert high > low


def test_risks_are_clamped_to_probability_range():
    profile = make_profile(
        uncertainty_level=1.0,
        mastery_confidence_mean=0.0,
        stall_risk_baseline=1.0,
        regression_risk_baseline=1.0,
        budget_multiplier=2.0,
    )
    assert 0.0 <= estimate_stall_risk(profile, "kc_x", weak_prereq_fraction=1.0, confidence=0.0, graph_depth=10) <= 1.0
    assert 0.0 <= estimate_regression_risk(profile, "kc_x", confidence=0.0) <= 1.0
