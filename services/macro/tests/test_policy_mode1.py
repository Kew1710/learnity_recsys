"""Тесты SubgraphQAgent и train_policy."""
import random
import pytest
from services.macro.policy_mode1 import (
    SubgraphQAgent,
    train_policy,
    _state_key,
    _discretize,
)

SUBGRAPH = {
    "nodes": ["A", "B", "C"],
    "edges": [
        {"from": "A", "to": "B", "strength": 0.9},
        {"from": "B", "to": "C", "strength": 0.8},
    ],
}


class TestDiscretize:
    def test_zero(self):
        assert _discretize(0.0) == 0

    def test_one(self):
        assert _discretize(1.0) == 4  # min(4, int(1.0 * 5)) = min(4, 5) = 4

    def test_midpoints(self):
        assert _discretize(0.19) == 0
        assert _discretize(0.20) == 1
        assert _discretize(0.39) == 1
        assert _discretize(0.40) == 2


class TestStateKey:
    def test_deterministic(self):
        mastery = {"A": 0.3, "B": 0.6, "C": 0.0}
        order = ["A", "B", "C"]
        k1 = _state_key(mastery, order)
        k2 = _state_key(mastery, order)
        assert k1 == k2

    def test_different_masteries_different_keys(self):
        order = ["A", "B"]
        k1 = _state_key({"A": 0.1, "B": 0.1}, order)
        k2 = _state_key({"A": 0.9, "B": 0.9}, order)
        assert k1 != k2

    def test_missing_kc_defaults_to_zero(self):
        k = _state_key({}, ["A"])
        assert k == (0,)

    def test_profile_features_extend_state_key(self):
        base = _state_key({"A": 0.2}, ["A"])
        enriched = _state_key(
            {"A": 0.2},
            ["A"],
            {
                "mean_confidence": 0.7,
                "weak_prereq_fraction": 0.5,
                "learning_speed_recent": 0.06,
                "stall_risk_baseline": 0.4,
                "pacing_mode": "careful",
            },
        )
        assert len(enriched) > len(base)


class TestSubgraphQAgent:
    def _make_agent(self) -> SubgraphQAgent:
        return SubgraphQAgent(
            node_order=["A", "B", "C"],
            rng=random.Random(42),
        )

    def test_select_action_returns_valid_action(self):
        agent = self._make_agent()
        action = agent.select_action({"A": 0.3, "B": 0.5, "C": 0.0}, ["A", "B", "C"])
        assert action in ["A", "B", "C"]

    def test_raises_on_empty_actions(self):
        agent = self._make_agent()
        with pytest.raises(ValueError):
            agent.select_action({}, [])

    def test_update_changes_q_values(self):
        agent = self._make_agent()
        state = {"A": 0.3, "B": 0.5, "C": 0.0}
        agent.update(state, "B", reward=1.0, next_state=state, available_next=["A", "B", "C"])
        agent.update(state, "B", reward=1.0, next_state=state, available_next=["A", "B", "C"])
        # После двух обновлений Q(s, B) должен быть > 0
        key = _state_key(state, ["A", "B", "C"])
        assert agent.q[key]["B"] > 0.0

    def test_exploitation_selects_best(self):
        agent = self._make_agent()
        state = {"A": 0.3, "B": 0.5, "C": 0.0}
        # Учим что C хорошая
        for _ in range(20):
            agent.update(state, "C", reward=1.0, next_state=state, available_next=["A", "B", "C"])
            agent.update(state, "A", reward=-1.0, next_state=state, available_next=["A", "B", "C"])
        # epsilon=0 → должен выбрать C
        action = agent.select_action(state, ["A", "B", "C"], epsilon=0.0)
        assert action == "C"

    def test_profile_features_change_q_bucket(self):
        agent = self._make_agent()
        state = {"A": 0.3, "B": 0.5, "C": 0.0}
        careful = {
            "mean_confidence": 0.2,
            "weak_prereq_fraction": 0.6,
            "learning_speed_recent": 0.02,
            "stall_risk_baseline": 0.7,
            "pacing_mode": "careful",
        }
        aggressive = {
            "mean_confidence": 0.9,
            "weak_prereq_fraction": 0.1,
            "learning_speed_recent": 0.08,
            "stall_risk_baseline": 0.1,
            "pacing_mode": "aggressive",
        }
        agent.update(state, "A", reward=1.0, next_state=state, available_next=["A"], profile_features=careful)
        agent.update(state, "B", reward=1.0, next_state=state, available_next=["B"], profile_features=aggressive)
        careful_key = _state_key(state, ["A", "B", "C"], careful)
        aggressive_key = _state_key(state, ["A", "B", "C"], aggressive)
        assert careful_key != aggressive_key
        assert agent.q[careful_key]["A"] > 0
        assert agent.q[aggressive_key]["B"] > 0


class TestTrainPolicy:
    def test_returns_agent(self):
        rng = random.Random(42)
        agent = train_policy(
            subgraph=SUBGRAPH,
            target_kc_id="C",
            initial_mastery={"A": 0.2, "B": 0.1, "C": 0.0},
            target_mastery=0.80,
            n_episodes=50,
            rng=rng,
        )
        assert isinstance(agent, SubgraphQAgent)

    def test_policy_path_uses_version_marker(self):
        from services.macro.policy_mode1 import policy_path
        assert "policy_v2_cluster_" in policy_path(1, "C")

    def test_q_table_populated_after_training(self):
        rng = random.Random(42)
        agent = train_policy(
            subgraph=SUBGRAPH,
            target_kc_id="C",
            initial_mastery={"A": 0.2, "B": 0.1, "C": 0.0},
            n_episodes=100,
            rng=rng,
        )
        # Q-таблица должна содержать записи
        assert len(agent.q) > 0

    def test_trained_agent_selects_action(self):
        rng = random.Random(7)
        agent = train_policy(
            subgraph=SUBGRAPH,
            target_kc_id="C",
            initial_mastery={"A": 0.3, "B": 0.2, "C": 0.0},
            n_episodes=100,
            rng=rng,
        )
        action = agent.select_action(
            {"A": 0.3, "B": 0.2, "C": 0.0},
            ["A", "B", "C"],
            epsilon=0.0,
        )
        assert action in ["A", "B", "C"]
