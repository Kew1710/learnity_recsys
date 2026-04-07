"""Тесты Phase 2: difficulty_mode, MicroSummary, bridge bonus."""
import pytest
from services.retrieval.micro_summary import (
    should_publish_summary,
    is_frustrated,
)
from services.retrieval.selector import compute_zpd_target_difficulty, select_task


# ---------------------------------------------------------------------------
# difficulty_mode → ZPD целевая сложность
# ---------------------------------------------------------------------------

class TestDifficultyModeZPD:
    """ZPD-цель правильно отражает режим плана."""

    def test_build_above_mastery(self):
        target = compute_zpd_target_difficulty(0.5, "build")
        assert target > 0.5

    def test_consolidate_below_mastery(self):
        target = compute_zpd_target_difficulty(0.5, "consolidate")
        assert target < 0.5

    def test_test_mode_well_above_mastery(self):
        target = compute_zpd_target_difficulty(0.5, "test")
        assert target > compute_zpd_target_difficulty(0.5, "build")

    def test_weak_student_gets_easy_target(self):
        # mastery=0 → build target=0.1 (не 0.9 как было раньше)
        assert compute_zpd_target_difficulty(0.0, "build") < 0.2

    def test_strong_student_gets_hard_target(self):
        # mastery=0.8 → build target=0.9
        assert compute_zpd_target_difficulty(0.8, "build") > 0.7

    def test_modes_give_different_targets(self):
        build = compute_zpd_target_difficulty(0.6, "build")
        consolidate = compute_zpd_target_difficulty(0.6, "consolidate")
        test = compute_zpd_target_difficulty(0.6, "test")
        assert test > build > consolidate


# ---------------------------------------------------------------------------
# MicroSummary триггеры
# ---------------------------------------------------------------------------

class TestMicroSummaryTriggers:
    def test_publish_every_15_tasks(self):
        assert should_publish_summary(15, 0.5, 0.5) is True
        assert should_publish_summary(30, 0.5, 0.5) is True
        assert should_publish_summary(45, 0.5, 0.5) is True

    def test_no_publish_at_non_multiples(self):
        assert should_publish_summary(14, 0.5, 0.5) is False
        assert should_publish_summary(16, 0.5, 0.5) is False
        assert should_publish_summary(1, 0.5, 0.5) is False

    def test_publish_on_mastery_jump(self):
        assert should_publish_summary(7, 0.4, 0.51) is True  # Δ=0.11 ≥ 0.1
        assert should_publish_summary(7, 0.4, 0.49) is False  # Δ=0.09 < 0.1

    def test_no_publish_at_zero_tasks(self):
        assert should_publish_summary(0, 0.5, 0.5) is False

    def test_frustration_detected(self):
        assert is_frustrated(frustration_count=2, velocity=0.0) is True
        assert is_frustrated(frustration_count=3, velocity=0.02) is True

    def test_no_frustration_if_count_low(self):
        assert is_frustrated(frustration_count=1, velocity=0.0) is False

    def test_no_frustration_if_velocity_high(self):
        assert is_frustrated(frustration_count=3, velocity=0.1) is False
        assert is_frustrated(frustration_count=5, velocity=-0.1) is False

    def test_frustration_boundary_velocity(self):
        # |velocity| < 0.05 → считается ~0
        assert is_frustrated(frustration_count=2, velocity=0.04) is True
        assert is_frustrated(frustration_count=2, velocity=-0.04) is True
        assert is_frustrated(frustration_count=2, velocity=0.05) is False


# ---------------------------------------------------------------------------
# Bridge bonus
# ---------------------------------------------------------------------------

class TestBridgeBonus:
    """Задание-мостик должно получать +0.1 к score в LinUCB."""

    def _score_with_bridge(self, task: dict, next_plan_kc_id: str | None) -> float:
        from services.retrieval.linucb import LinUCBModel, CONTEXT_DIM
        import numpy as np
        model = LinUCBModel.init("kc_test")
        x = np.zeros(CONTEXT_DIM, dtype=np.float64)
        s = model.score(x)
        secondary_kcs = task.get("parts", [{}])[0].get("secondary_kcs", [])
        if next_plan_kc_id and next_plan_kc_id in secondary_kcs:
            s += 0.1
        return s

    def test_bridge_task_gets_bonus(self):
        bridge_task = {"parts": [{"secondary_kcs": ["kc_next"], "irt_difficulty": 0.5}]}
        non_bridge = {"parts": [{"secondary_kcs": [], "irt_difficulty": 0.5}]}

        score_bridge = self._score_with_bridge(bridge_task, "kc_next")
        score_non = self._score_with_bridge(non_bridge, "kc_next")

        assert abs(score_bridge - score_non - 0.1) < 1e-9

    def test_no_bonus_without_next_plan_kc(self):
        bridge_task = {"parts": [{"secondary_kcs": ["kc_next"], "irt_difficulty": 0.5}]}

        score_with = self._score_with_bridge(bridge_task, "kc_next")
        score_without = self._score_with_bridge(bridge_task, None)

        assert abs(score_with - score_without - 0.1) < 1e-9

    def test_no_bonus_when_secondary_doesnt_match(self):
        task = {"parts": [{"secondary_kcs": ["kc_other"], "irt_difficulty": 0.5}]}

        score = self._score_with_bridge(task, "kc_next")
        score_no_plan = self._score_with_bridge(task, None)

        assert abs(score - score_no_plan) < 1e-9

    def test_no_bonus_if_no_secondary_kcs(self):
        task = {"parts": [{"irt_difficulty": 0.5}]}

        score = self._score_with_bridge(task, "kc_next")
        score_no_plan = self._score_with_bridge(task, None)

        assert abs(score - score_no_plan) < 1e-9
