"""Tests for macro step outcome logging helper."""

from __future__ import annotations

import uuid

import pytest

import services.macro.outcomes as outcomes


@pytest.mark.asyncio
async def test_log_macro_step_outcome_writes_row(monkeypatch):
    executed: list[tuple[str, dict]] = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, statement, params):
            executed.append((str(statement), params))
            return None

        async def commit(self):
            return None

    monkeypatch.setattr(outcomes, "AsyncSessionLocal", lambda: FakeSession())

    await outcomes.log_macro_step_outcome(
        student_id=uuid.uuid4(),
        plan_id=uuid.uuid4(),
        kc_id="kc_target",
        outcome_type="completed",
        tasks_spent=12,
    )

    assert executed
    assert "INSERT INTO macro_step_outcomes" in executed[0][0]
    assert executed[0][1]["outcome_type"] == "completed"
