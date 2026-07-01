"""Tests for MacroStudentProfile builder helpers."""

from __future__ import annotations

import uuid

from services.macro.profile_builder import (
    build_profile_from_inputs,
    derive_budget_multiplier,
    derive_pacing_mode,
    derive_prereq_strictness,
    derive_step_granularity,
    derive_test_readiness_bias,
)


def test_derive_pacing_mode_prefers_careful_for_high_uncertainty():
    result = derive_pacing_mode(
        uncertainty_level=0.7,
        frustration_risk=0.2,
        learning_speed_recent=0.08,
        recovery_after_error=0.07,
    )
    assert result == "careful"


def test_derive_pacing_mode_prefers_aggressive_for_fast_recovery():
    result = derive_pacing_mode(
        uncertainty_level=0.2,
        frustration_risk=0.1,
        learning_speed_recent=0.08,
        recovery_after_error=0.06,
    )
    assert result == "aggressive"


def test_budget_multiplier_increases_with_risk():
    low = derive_budget_multiplier("balanced", uncertainty_level=0.1, weak_prereq_fraction=0.1)
    high = derive_budget_multiplier("careful", uncertainty_level=0.6, weak_prereq_fraction=0.5)
    assert high > low


def test_profile_builder_marks_weak_prereqs_and_careful_mode():
    student_id = uuid.uuid4()
    detailed_mastery = {
        "kc_target": {"probability_effective": 0.4, "confidence": 0.2, "attempts_count": 2},
        "kc_prereq": {"probability_effective": 0.35, "confidence": 0.25, "attempts_count": 1},
        "kc_extra": {"probability_effective": 0.8, "confidence": 0.8, "attempts_count": 8},
    }

    profile = build_profile_from_inputs(
        student_id,
        target_kc_id="kc_target",
        detailed_mastery=detailed_mastery,
        target_subgraph_nodes=["kc_target", "kc_prereq"],
        learning_speed_global=0.02,
        learning_speed_recent=0.01,
        frustration_risk=0.5,
        recovery_after_error=0.01,
    )

    assert profile.target_kc_id == "kc_target"
    assert profile.weak_prereq_fraction >= 0.5
    assert profile.uncertainty_level > 0.0
    assert profile.pacing_mode == "careful"
    assert profile.budget_multiplier > 1.0
    assert profile.prereq_strictness > 0.5


def test_profile_builder_marks_fast_student_aggressive():
    student_id = uuid.uuid4()
    detailed_mastery = {
        "kc_target": {"probability_effective": 0.65, "confidence": 0.8, "attempts_count": 6},
        "kc_prereq": {"probability_effective": 0.75, "confidence": 0.85, "attempts_count": 9},
    }

    profile = build_profile_from_inputs(
        student_id,
        target_kc_id="kc_target",
        detailed_mastery=detailed_mastery,
        target_subgraph_nodes=["kc_target", "kc_prereq"],
        learning_speed_global=0.07,
        learning_speed_recent=0.08,
        frustration_risk=0.05,
        recovery_after_error=0.07,
    )

    assert profile.pacing_mode == "aggressive"
    assert profile.budget_multiplier < 1.1
    assert profile.test_readiness_bias > 0.6
    assert profile.step_granularity >= 1.0


def test_derived_values_stay_in_reasonable_ranges():
    assert 0.0 <= derive_prereq_strictness(0.3, 0.4) <= 1.0
    assert 0.0 <= derive_test_readiness_bias(0.7, 0.05, 0.2) <= 1.0
    assert derive_step_granularity("balanced", 0.2, 0.1) >= 0.5
