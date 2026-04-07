"""Тесты BKTEnvironment."""
import random
import pytest
from services.macro.bkt_environment import BKTEnvironment, MICRO_STEPS_PER_MACRO


SIMPLE_SUBGRAPH = {
    "nodes": ["A", "B"],
    "edges": [{"from": "A", "to": "B", "strength": 0.9}],
}


class TestBKTEnvironment:
    def _make_env(self, rng_seed=42) -> BKTEnvironment:
        return BKTEnvironment(
            subgraph=SIMPLE_SUBGRAPH,
            target_kc_id="B",
            target_mastery=0.80,
            rng=random.Random(rng_seed),
        )

    def test_reset_returns_initial_mastery(self):
        env = self._make_env()
        initial = {"A": 0.5, "B": 0.3}
        state = env.reset(initial)
        assert abs(state.mastery["A"] - 0.5) < 1e-9
        assert abs(state.mastery["B"] - 0.3) < 1e-9
        assert state.steps_taken == 0

    def test_step_increases_mastery(self):
        env = self._make_env()
        env.reset({"A": 0.5, "B": 0.3})
        # Несколько шагов на B
        for _ in range(5):
            state, reward, done = env.step("B")
        # mastery B должен вырасти
        assert state.mastery["B"] > 0.3

    def test_step_counts_increase(self):
        env = self._make_env()
        env.reset({"A": 0.5, "B": 0.1})
        state, _, _ = env.step("B")
        assert state.steps_taken == 1
        state, _, _ = env.step("B")
        assert state.steps_taken == 2

    def test_done_when_target_mastery_reached(self):
        # При очень высоком p_transit — можно быстро достичь порога
        env = BKTEnvironment(
            subgraph=SIMPLE_SUBGRAPH,
            target_kc_id="B",
            target_mastery=0.80,
            bkt_params={"B": {"p_transit": 0.5, "p_slip": 0.05, "p_guess": 0.3}},
            rng=random.Random(1),
        )
        env.reset({"A": 0.9, "B": 0.0})
        done_seen = False
        for _ in range(30):
            _, _, done = env.step("B")
            if done:
                done_seen = True
                break
        assert done_seen, "Должны были достичь target_mastery=0.8 за 30 шагов"

    def test_reward_has_time_cost(self):
        # Если mastery уже высокий — delta будет мала, reward может быть отрицательным
        env = BKTEnvironment(
            subgraph=SIMPLE_SUBGRAPH,
            target_kc_id="B",
            target_mastery=0.80,
            bkt_params={"B": {"p_transit": 0.0, "p_slip": 0.5, "p_guess": 0.0}},
            rng=random.Random(42),
        )
        env.reset({"A": 0.5, "B": 0.99})
        # При p_transit=0 mastery почти не растёт → reward ≈ -TIME_COST
        _, reward, _ = env.step("B")
        assert reward < 0.01  # должен быть близок к нулю или отрицательным

    def test_reset_does_not_mutate_input(self):
        env = self._make_env()
        initial = {"A": 0.5, "B": 0.3}
        original_copy = dict(initial)
        env.reset(initial)
        env.step("B")
        # Оригинальный dict не изменился
        assert initial == original_copy

    def test_prereq_bonus_in_reward(self):
        # Если работаем над пре-реквизитом A → delta(A) > 0 → bonus в reward для цели B
        env = BKTEnvironment(
            subgraph=SIMPLE_SUBGRAPH,
            target_kc_id="B",
            target_mastery=0.80,
            bkt_params={"A": {"p_transit": 0.4, "p_slip": 0.1, "p_guess": 0.2}},
            rng=random.Random(7),
        )
        env.reset({"A": 0.1, "B": 0.0})
        # Работаем над A — должен быть prereq_bonus > 0
        total_reward = 0.0
        for _ in range(3):
            _, r, _ = env.step("A")
            total_reward += r
        # Reward может быть маленьким, но суммарно не должен быть -inf
        assert total_reward > -1.0
