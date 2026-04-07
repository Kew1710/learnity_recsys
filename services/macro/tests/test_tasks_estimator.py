"""Тесты TasksToMasteryEstimator."""
import random
import pytest
from services.macro.tasks_estimator import simulate_tasks_to_mastery, estimate


class TestSimulateTasksToMastery:
    def test_already_mastered_returns_zero(self):
        # Если m_current >= m_target — нужно 0 задач
        result = simulate_tasks_to_mastery(
            m_current=0.9, m_target=0.8, n_sims=100, rng=random.Random(42)
        )
        assert result == 0

    def test_larger_gap_needs_more_tasks(self):
        rng = random.Random(42)
        small_gap = simulate_tasks_to_mastery(0.7, 0.8, n_sims=200, rng=rng)
        large_gap = simulate_tasks_to_mastery(0.1, 0.8, n_sims=200, rng=rng)
        assert large_gap > small_gap

    def test_higher_transit_needs_fewer_tasks(self):
        rng = random.Random(42)
        slow = simulate_tasks_to_mastery(0.3, 0.8, p_transit=0.05, n_sims=200, rng=rng)
        fast = simulate_tasks_to_mastery(0.3, 0.8, p_transit=0.3, n_sims=200, rng=rng)
        assert fast < slow

    def test_returns_positive_integer(self):
        result = simulate_tasks_to_mastery(0.3, 0.8, n_sims=50, rng=random.Random(1))
        assert isinstance(result, int)
        assert result >= 0

    def test_deterministic_with_seed(self):
        r1 = simulate_tasks_to_mastery(0.3, 0.8, n_sims=100, rng=random.Random(99))
        r2 = simulate_tasks_to_mastery(0.3, 0.8, n_sims=100, rng=random.Random(99))
        assert r1 == r2


class TestEstimate:
    def test_no_cluster_avg_uses_simulation(self):
        result = estimate(
            m_current=0.3, m_target=0.8,
            cluster_avg=None,
            n_practiced_in_subject=0,
            n_sims=100, rng=random.Random(42),
        )
        assert result > 0

    def test_full_personal_history_uses_cluster_avg(self):
        # n_practiced=20 → α=1.0 → result = cluster_avg
        result = estimate(
            m_current=0.3, m_target=0.8,
            cluster_avg=10.0,
            n_practiced_in_subject=20,
            n_sims=100, rng=random.Random(42),
        )
        assert result == 10

    def test_zero_history_uses_simulation(self):
        # n_practiced=0 → α=0.0 → result = sim estimate
        sim = simulate_tasks_to_mastery(0.3, 0.8, n_sims=100, rng=random.Random(42))
        est = estimate(
            m_current=0.3, m_target=0.8,
            cluster_avg=999.0,  # не должно использоваться
            n_practiced_in_subject=0,
            n_sims=100, rng=random.Random(42),
        )
        assert est == sim

    def test_blending(self):
        # n_practiced=10 → α=0.5 → blend(5, sim)
        result = estimate(
            m_current=0.3, m_target=0.8,
            cluster_avg=5.0,
            n_practiced_in_subject=10,
            n_sims=100, rng=random.Random(42),
        )
        assert result >= 1

    def test_result_at_least_one(self):
        result = estimate(
            m_current=0.79, m_target=0.80,
            cluster_avg=1.0,
            n_practiced_in_subject=20,
            n_sims=100, rng=random.Random(1),
        )
        assert result >= 1
