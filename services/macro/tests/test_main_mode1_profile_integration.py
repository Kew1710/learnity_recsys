"""Focused tests for mode1 rollout with MacroStudentProfile."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from services.macro.main import _rollout_plan
from services.macro.policy_mode1 import SubgraphQAgent
from services.macro.student_profile import MacroStudentProfile


class StaticAgent(SubgraphQAgent):
    def __init__(self, action_order: list[str]):
        super().__init__(node_order=action_order)
        self._action_order = action_order

    def select_action(self, state, available_actions, epsilon=0.1, profile_features=None):
        for action in self._action_order:
            if action in available_actions:
                return action
        return available_actions[0]


def make_profile(*, pacing_mode: str, budget_multiplier: float, mastery_confidence_mean: float) -> MacroStudentProfile:
    return MacroStudentProfile(
        student_id=uuid.uuid4(),
        version=1,
        updated_at=datetime.now(timezone.utc),
        target_kc_id="C",
        uncertainty_level=0.2,
        mastery_confidence_mean=mastery_confidence_mean,
        weak_prereq_fraction=0.3,
        target_subgraph_mastery_mean=0.4,
        learning_speed_global=0.05,
        learning_speed_recent=0.05,
        tasks_to_gain_01_mastery=2.0,
        recovery_after_error=0.04,
        frustration_risk=0.1,
        stall_risk_baseline=0.25,
        regression_risk_baseline=0.2,
        pacing_mode=pacing_mode,
        budget_multiplier=budget_multiplier,
        prereq_strictness=0.5,
        test_readiness_bias=0.6,
        step_granularity=1.0,
    )


def test_rollout_plan_budget_depends_on_profile():
    agent = StaticAgent(["A", "B"])
    subgraph = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"from": "A", "to": "B", "strength": 1.0},
            {"from": "B", "to": "C", "strength": 1.0},
        ],
    }
    mastery = {"A": 0.3, "B": 0.3, "C": 0.3}
    detailed_mastery = {
        "A": {"confidence": 0.8},
        "B": {"confidence": 0.8},
        "C": {"confidence": 0.8},
    }

    class Req:
        target_kc_id = "C"
        mastery_threshold = 0.8

    low = _rollout_plan(
        agent,
        subgraph,
        mastery,
        detailed_mastery,
        Req,
        8,
        {},
        make_profile(pacing_mode="aggressive", budget_multiplier=0.85, mastery_confidence_mean=0.8),
        {"mean_confidence": 0.8, "weak_prereq_fraction": 0.2, "learning_speed_recent": 0.06, "stall_risk_baseline": 0.2, "pacing_mode": "aggressive"},
    )
    high = _rollout_plan(
        agent,
        subgraph,
        mastery,
        detailed_mastery,
        Req,
        8,
        {},
        make_profile(pacing_mode="careful", budget_multiplier=1.35, mastery_confidence_mean=0.4),
        {"mean_confidence": 0.4, "weak_prereq_fraction": 0.5, "learning_speed_recent": 0.02, "stall_risk_baseline": 0.5, "pacing_mode": "careful"},
    )

    assert high[0].tasks_budget > low[0].tasks_budget
    assert "pacing=careful" in high[0].reason
