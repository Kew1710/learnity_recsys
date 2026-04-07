"""Юнит-тесты LinUCBModel. Без HTTP и БД."""

import numpy as np
import pytest
from services.retrieval.linucb import LinUCBModel, CONTEXT_DIM


def make_x(val: float = 0.5) -> np.ndarray:
    return np.full(CONTEXT_DIM, val, dtype=np.float64)


class TestLinUCBModel:
    def test_init_identity_matrix(self):
        m = LinUCBModel.init("kc_test")
        assert m.A.shape == (CONTEXT_DIM, CONTEXT_DIM)
        assert np.allclose(m.A, np.eye(CONTEXT_DIM))
        assert np.allclose(m.b, np.zeros(CONTEXT_DIM))

    def test_score_returns_float(self):
        m = LinUCBModel.init("kc_test")
        score = m.score(make_x(0.5))
        assert isinstance(score, float)

    def test_score_increases_after_positive_reward(self):
        m = LinUCBModel.init("kc_test")
        x = make_x(0.5)
        score_before = m.score(x)
        m.update(x, reward=1.0)
        score_after = m.score(x)
        # После положительной награды θ растёт → score должен вырасти
        assert score_after > score_before

    def test_ucb_bonus_decreases_after_observation(self):
        """UCB-бонус √(xᵀA⁻¹x) должен снижаться по мере накопления данных."""
        m = LinUCBModel.init("kc_test")
        x = make_x(0.5)

        def ucb_bonus() -> float:
            A_inv = np.linalg.inv(m.A)
            return float(np.sqrt(max(0.0, x @ A_inv @ x)))

        bonus_before = ucb_bonus()
        for _ in range(10):
            m.update(x, reward=0.0)
        bonus_after = ucb_bonus()
        assert bonus_after < bonus_before

    def test_serialize_roundtrip(self):
        m = LinUCBModel.init("kc_alg")
        x = make_x(0.7)
        m.update(x, reward=0.5)
        m.update(make_x(0.3), reward=-0.1)

        a_bytes, b_bytes = m.to_bytes()
        m2 = LinUCBModel.from_bytes("kc_alg", 0, a_bytes, b_bytes)

        assert np.allclose(m.A, m2.A)
        assert np.allclose(m.b, m2.b)
        assert np.isclose(m.score(x), m2.score(x))

    def test_different_tasks_get_different_scores(self):
        m = LinUCBModel.init("kc_test")
        # Обучаем на задании 1 с высокой наградой
        x1 = np.zeros(CONTEXT_DIM, dtype=np.float64)
        x1[0] = 1.0
        x2 = np.zeros(CONTEXT_DIM, dtype=np.float64)
        x2[0] = 0.0

        for _ in range(5):
            m.update(x1, reward=1.0)
            m.update(x2, reward=0.0)

        assert m.score(x1) != m.score(x2)

    def test_score_reuses_inverse_cache(self, monkeypatch):
        m = LinUCBModel.init("kc_test")
        x = make_x(0.5)

        calls = 0
        original_inv = np.linalg.inv

        def wrapped_inv(matrix):
            nonlocal calls
            calls += 1
            return original_inv(matrix)

        monkeypatch.setattr(np.linalg, "inv", wrapped_inv)

        _ = m.score(x)
        _ = m.score(x)

        assert calls == 1

        # После update inverse поддерживается инкрементально,
        # поэтому повторный score не требует нового inv().
        m.update(x, reward=0.3)
        _ = m.score(x)
        assert calls == 1
