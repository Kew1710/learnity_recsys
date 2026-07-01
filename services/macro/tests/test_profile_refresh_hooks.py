"""Tests for MacroStudentProfile refresh hooks in lifecycle."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

import services.macro.plan_lifecycle as lifecycle


@pytest.mark.asyncio
async def test_advance_step_triggers_profile_refresh(monkeypatch):
    student_id = uuid.uuid4()

    class FakeResult:
        def __init__(self, row=None):
            self._row = row

        def fetchone(self):
            return self._row

    class FakeSession:
        def __init__(self):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, statement, params):
            self.calls += 1
            if self.calls == 1:
                return FakeResult((uuid.uuid4(), uuid.uuid4(), "kc_target", 20, 12, "build"))
            if "RETURNING id" in str(statement):
                return FakeResult((uuid.uuid4(),))
            return FakeResult()

        async def commit(self):
            return None

    refresh_mock = AsyncMock()
    outcome_mock = AsyncMock()
    monkeypatch.setattr(lifecycle, "AsyncSessionLocal", lambda: FakeSession())
    monkeypatch.setattr(lifecycle, "refresh_macro_profile", refresh_mock)
    monkeypatch.setattr(lifecycle, "get_macro_student_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(lifecycle, "log_macro_step_outcome", outcome_mock)

    advanced = await lifecycle.advance_step(student_id)

    assert advanced is True
    refresh_mock.assert_awaited_once_with(student_id)
    outcome_mock.assert_awaited_once()
