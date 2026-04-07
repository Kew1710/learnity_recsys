"""Тесты lifecycle-логики в macro Kafka consumer."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from services.macro.plan_lifecycle import PlanAction
import services.macro.kafka_consumer as kc


@pytest.mark.asyncio
async def test_handle_summary_applies_actions_and_creates_replan_alert(monkeypatch):
    student_id = uuid.uuid4()
    plan_id = uuid.uuid4()
    summary = {
        "student_id": str(student_id),
        "kc_id": "kc_target",
        "mastery_current": 0.42,
        "tasks_spent": 12,
        "velocity": -0.08,
        "avg_score": 0.30,
        "frustration_count": 2,
    }

    monkeypatch.setattr(kc, "check_and_advance", AsyncMock(return_value=False))
    monkeypatch.setattr(kc, "_get_active_plan_step", AsyncMock(return_value=(plan_id, "kc_target", 20)))
    monkeypatch.setattr(kc, "_find_weakest_prereq", AsyncMock(return_value="kc_prereq"))

    actions = [
        PlanAction("set_difficulty_mode", {"kc_id": "kc_target", "difficulty_mode": "consolidate"}),
        PlanAction("replan", {"reason": "velocity=-0.080 < -0.05"}),
    ]
    monkeypatch.setattr(kc, "evaluate_micro_summary", lambda *_args, **_kwargs: actions)

    apply_mock = AsyncMock()
    replan_alert_mock = AsyncMock()
    plateau_alert_mock = AsyncMock()
    monkeypatch.setattr(kc, "apply_plan_actions", apply_mock)
    monkeypatch.setattr(kc, "_create_replan_alert", replan_alert_mock)
    monkeypatch.setattr(kc, "create_plateau_alert", plateau_alert_mock)

    await kc._handle_summary(summary)

    apply_mock.assert_awaited_once()
    apply_args = apply_mock.await_args.args
    assert apply_args[0] == student_id
    assert len(apply_args[1]) == 1
    assert apply_args[1][0].action_type == "set_difficulty_mode"

    replan_alert_mock.assert_awaited_once()
    plateau_alert_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_summary_short_circuits_when_advanced(monkeypatch):
    student_id = uuid.uuid4()
    summary = {
        "student_id": str(student_id),
        "kc_id": "kc_target",
        "mastery_current": 0.9,
        "tasks_spent": 20,
        "velocity": 0.05,
        "avg_score": 0.8,
    }

    monkeypatch.setattr(kc, "check_and_advance", AsyncMock(return_value=True))
    get_active_mock = AsyncMock()
    monkeypatch.setattr(kc, "_get_active_plan_step", get_active_mock)

    await kc._handle_summary(summary)

    get_active_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_summary_ignores_non_active_kc(monkeypatch):
    student_id = uuid.uuid4()
    plan_id = uuid.uuid4()
    summary = {
        "student_id": str(student_id),
        "kc_id": "kc_other",
        "mastery_current": 0.4,
        "tasks_spent": 8,
        "velocity": 0.0,
        "avg_score": 0.4,
    }

    monkeypatch.setattr(kc, "check_and_advance", AsyncMock(return_value=False))
    monkeypatch.setattr(kc, "_get_active_plan_step", AsyncMock(return_value=(plan_id, "kc_target", 15)))

    apply_mock = AsyncMock()
    monkeypatch.setattr(kc, "apply_plan_actions", apply_mock)

    await kc._handle_summary(summary)

    apply_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_summary_creates_plateau_alert(monkeypatch):
    student_id = uuid.uuid4()
    plan_id = uuid.uuid4()
    summary = {
        "student_id": str(student_id),
        "kc_id": "kc_target",
        "mastery_current": 0.51,
        "tasks_spent": 55,
        "velocity": 0.0,
        "avg_score": 0.45,
        "frustration_count": 0,
    }

    monkeypatch.setattr(kc, "check_and_advance", AsyncMock(return_value=False))
    monkeypatch.setattr(kc, "_get_active_plan_step", AsyncMock(return_value=(plan_id, "kc_target", 100)))
    monkeypatch.setattr(kc, "_find_weakest_prereq", AsyncMock(return_value=None))
    monkeypatch.setattr(kc, "evaluate_micro_summary", lambda *_args, **_kwargs: [])

    plateau_alert_mock = AsyncMock()
    monkeypatch.setattr(kc, "create_plateau_alert", plateau_alert_mock)

    await kc._handle_summary(summary)

    plateau_alert_mock.assert_awaited_once_with(student_id, plan_id, "kc_target", 0.51, 55)
