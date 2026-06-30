"""Unit tests for ThompsonModel. No HTTP or DB."""

import numpy as np
import pytest
from services.retrieval.thompson import ThompsonModel, CONTEXT_DIM


def make_x(val: float = 0.5) -> np.ndarray:
    return np.full(CONTEXT_DIM, val, dtype=np.float64)


class TestThompsonModel:
    def test_init_identity_matrix(self):
        m = ThompsonModel.init("kc_test")
        assert m.A.shape == (CONTEXT_DIM, CONTEXT_DIM)
        assert np.allclose(m.A, np.eye(CONTEXT_DIM))
        assert np.allclose(m.b, np.zeros(CONTEXT_DIM))

    def test_score_returns_float(self):
        m = ThompsonModel.init("kc_test")
        score = m.score(make_x(0.5))
        assert isinstance(score, float)

    def test_score_is_stochastic(self):
        m = ThompsonModel.init("kc_test")
        x = make_x(0.5)
        scores = [m.score(x) for _ in range(50)]
        assert len(set(scores)) > 1

    def test_score_exploit_is_deterministic(self):
        m = ThompsonModel.init("kc_test")
        x = make_x(0.5)
        s1 = m.score_exploit(x)
        s2 = m.score_exploit(x)
        assert s1 == s2

    def test_mean_score_shifts_after_positive_reward(self):
        m = ThompsonModel.init("kc_test")
        x = make_x(0.5)
        exploit_before = m.score_exploit(x)
        for _ in range(10):
            m.update(x, reward=1.0)
        exploit_after = m.score_exploit(x)
        assert exploit_after > exploit_before

    def test_variance_decreases_after_observations(self):
        m = ThompsonModel.init("kc_test")
        x = make_x(0.5)

        np.random.seed(42)
        scores_before = [m.score(x) for _ in range(200)]
        var_before = np.var(scores_before)

        for _ in range(20):
            m.update(x, reward=0.5)

        np.random.seed(42)
        scores_after = [m.score(x) for _ in range(200)]
        var_after = np.var(scores_after)

        assert var_after < var_before

    def test_serialize_roundtrip(self):
        m = ThompsonModel.init("kc_alg")
        x = make_x(0.7)
        m.update(x, reward=0.5)
        m.update(make_x(0.3), reward=-0.1)

        a_bytes, b_bytes = m.to_bytes()
        m2 = ThompsonModel.from_bytes("kc_alg", 0, a_bytes, b_bytes)

        assert np.allclose(m.A, m2.A)
        assert np.allclose(m.b, m2.b)
        assert np.isclose(m.score_exploit(x), m2.score_exploit(x))

    def test_update_is_identical_to_linucb(self):
        from services.retrieval.linucb import LinUCBModel
        ts = ThompsonModel.init("kc_test")
        ucb = LinUCBModel.init("kc_test")
        x = make_x(0.6)

        for r in [0.5, -0.2, 1.0, 0.3]:
            ts.update(x, r)
            ucb.update(x, r)

        assert np.allclose(ts.A, ucb.A)
        assert np.allclose(ts.b, ucb.b)
