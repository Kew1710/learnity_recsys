"""Тесты LearningSpeedModel (чистая функция blend_speed)."""
import pytest
from services.macro.learning_speed import blend_speed, DEFAULT_SPEED, PERSONAL_HISTORY_FULL


class TestBlendSpeed:
    def test_no_data_returns_default(self):
        result = blend_speed(personal_speed=0.0, n_personal=0, cluster_speed=None)
        assert result == DEFAULT_SPEED

    def test_only_personal_no_cluster(self):
        result = blend_speed(personal_speed=0.08, n_personal=5, cluster_speed=None)
        assert result == 0.08

    def test_full_history_uses_personal_only(self):
        # n_personal >= PERSONAL_HISTORY_FULL → α=1.0 → result = personal_speed
        result = blend_speed(
            personal_speed=0.09,
            n_personal=PERSONAL_HISTORY_FULL,
            cluster_speed=0.03,
        )
        assert abs(result - 0.09) < 1e-9

    def test_zero_history_uses_cluster(self):
        # n_personal=0 → α=0.0 → result = cluster_speed
        result = blend_speed(
            personal_speed=0.0,
            n_personal=0,
            cluster_speed=0.06,
        )
        assert abs(result - 0.06) < 1e-9

    def test_half_history_blends(self):
        # n_personal=10 → α=0.5 → result = 0.5 * personal + 0.5 * cluster
        half = PERSONAL_HISTORY_FULL // 2
        result = blend_speed(
            personal_speed=0.10,
            n_personal=half,
            cluster_speed=0.04,
        )
        assert abs(result - 0.07) < 1e-9

    def test_more_personal_history_weights_personal_higher(self):
        low_alpha = blend_speed(0.10, n_personal=5, cluster_speed=0.02)
        high_alpha = blend_speed(0.10, n_personal=15, cluster_speed=0.02)
        assert high_alpha > low_alpha

    def test_alpha_clamped_at_one(self):
        # n_personal > PERSONAL_HISTORY_FULL → α не больше 1
        result = blend_speed(0.08, n_personal=100, cluster_speed=0.01)
        assert abs(result - 0.08) < 1e-9
