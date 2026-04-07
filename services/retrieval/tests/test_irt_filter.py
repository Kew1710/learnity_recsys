"""Юнит-тесты IRT-функций и ZPD-целевой сложности."""
import math
import pytest
from services.retrieval.selector import compute_p_correct, compute_zpd_target_difficulty


def make_task(irt_difficulty: float) -> dict:
    return {"parts": [{"irt_difficulty": irt_difficulty}]}


class TestComputePCorrect:
    def test_symmetric_at_zero(self):
        # mastery=0.5 → theta=0, difficulty=0 → P=0.5
        assert abs(compute_p_correct(0.5, 0.0) - 0.5) < 0.01

    def test_easy_task_high_mastery(self):
        # высокая mastery + лёгкое задание → P близка к 1
        assert compute_p_correct(0.95, -2.0) > 0.90

    def test_hard_task_low_mastery(self):
        # низкая mastery + сложное задание → P близка к 0
        assert compute_p_correct(0.05, 2.0) < 0.10

    def test_mastery_clamp_low(self):
        # mastery=0 не должна вызывать ZeroDivisionError
        p = compute_p_correct(0.0, 0.0)
        assert 0.0 <= p <= 1.0

    def test_mastery_clamp_high(self):
        # mastery=1 не должна вызывать ZeroDivisionError
        p = compute_p_correct(1.0, 0.0)
        assert 0.0 <= p <= 1.0

    def test_monotone_in_mastery(self):
        # при фиксированной сложности: больше mastery → больше P(correct)
        p_low = compute_p_correct(0.2, 0.0)
        p_mid = compute_p_correct(0.5, 0.0)
        p_high = compute_p_correct(0.8, 0.0)
        assert p_low < p_mid < p_high

    def test_monotone_in_difficulty(self):
        # при фиксированной mastery: труднее задание → меньше P(correct)
        p_easy = compute_p_correct(0.5, -1.0)
        p_mid = compute_p_correct(0.5, 0.0)
        p_hard = compute_p_correct(0.5, 1.0)
        assert p_easy > p_mid > p_hard

    def test_known_value(self):
        # mastery=0.73 → theta≈logit(0.73)≈0.989
        # difficulty=0.0 → P = sigmoid(0.989) ≈ 0.729
        p = compute_p_correct(0.73, 0.0)
        assert abs(p - 0.729) < 0.01


class TestComputeZPDTargetDifficulty:
    def test_build_above_mastery(self):
        assert abs(compute_zpd_target_difficulty(0.5, "build") - 0.6) < 1e-9

    def test_consolidate_below_mastery(self):
        assert abs(compute_zpd_target_difficulty(0.5, "consolidate") - 0.4) < 1e-9

    def test_test_mode_well_above_mastery(self):
        assert abs(compute_zpd_target_difficulty(0.5, "test") - 0.8) < 1e-9

    def test_unknown_mode_defaults_to_build(self):
        assert abs(compute_zpd_target_difficulty(0.5, "unknown") - 0.6) < 1e-9

    def test_clamp_min_at_low_mastery(self):
        # mastery=0, consolidate → 0 - 0.1 = -0.1 → clamp to 0.05
        assert compute_zpd_target_difficulty(0.0, "consolidate") == 0.05

    def test_clamp_max_at_high_mastery(self):
        # mastery=0.9, test → 0.9 + 0.3 = 1.2 → clamp to 0.95
        assert compute_zpd_target_difficulty(0.9, "test") == 0.95

    def test_weak_student_gets_easy_tasks(self):
        # mastery=0, build → target=0.1 (лёгкие задания, а не сложные)
        assert compute_zpd_target_difficulty(0.0, "build") == 0.1

    def test_strong_student_gets_hard_tasks(self):
        # mastery=0.8, build → target=0.9
        assert abs(compute_zpd_target_difficulty(0.8, "build") - 0.9) < 1e-9

    def test_modes_give_different_targets(self):
        build = compute_zpd_target_difficulty(0.5, "build")
        consolidate = compute_zpd_target_difficulty(0.5, "consolidate")
        test = compute_zpd_target_difficulty(0.5, "test")
        assert build != consolidate
        assert build != test
        assert test > build > consolidate
