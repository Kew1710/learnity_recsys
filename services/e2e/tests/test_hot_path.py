"""Stateful end-to-end hot-path tests for gateway orchestration."""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.gateway import main
from services.gateway.main import app
from services.profile.bkt import update_mastery


@dataclass
class StudentState:
    grade: int
    review_mode: bool = False
    mastery: dict[str, float] = field(default_factory=dict)
    consecutive_errors: dict[str, int] = field(default_factory=dict)
    consecutive_correct: dict[str, int] = field(default_factory=dict)


@pytest.fixture
async def client(monkeypatch):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        state = {
            "students": {
                str(uuid.UUID("00000000-0000-0000-0000-000000000001")): StudentState(grade=8),
            },
            "tasks": [
                {
                    "task_id": str(uuid.UUID("00000000-0000-0000-0000-000000000101")),
                    "kc_id": "kc_linear_eq_1var",
                    "subject": "algebra",
                    "recommendation_source": "phase1_heuristic",
                    "half_life_days": 45.0,
                    "task": {
                        "task_id": str(uuid.UUID("00000000-0000-0000-0000-000000000101")),
                        "parts": [{"irt_difficulty": 0.1, "primary_kcs": ["kc_linear_eq_1var"]}],
                    },
                },
                {
                    "task_id": str(uuid.UUID("00000000-0000-0000-0000-000000000102")),
                    "kc_id": "kc_linear_eq_1var",
                    "subject": "algebra",
                    "recommendation_source": "phase1_heuristic",
                    "half_life_days": 45.0,
                    "task": {
                        "task_id": str(uuid.UUID("00000000-0000-0000-0000-000000000102")),
                        "parts": [{"irt_difficulty": 0.3, "primary_kcs": ["kc_linear_eq_1var"]}],
                    },
                },
            ],
            "seen_task_ids": set(),
            "reward_updates": [],
        }

        async def submit_interaction(_http, student_id, payload):
            student = state["students"][str(student_id)]
            updated_mastery = {}
            mastery_before = {}
            consecutive_errors = {}
            for kc_id in payload["primary_kcs"]:
                current = student.mastery.get(kc_id, 0.0)
                mastery_before[kc_id] = current
                updated = update_mastery(
                    current_probability=current,
                    last_practiced=datetime.utcnow(),
                    score=payload["score"],
                    hints_used=payload.get("hints_used", 0),
                    p_transit=payload.get("p_transit", 0.1),
                    p_slip=payload.get("p_slip", 0.1),
                    p_guess=payload.get("p_guess", 0.2),
                    kc_role="primary",
                    review_mode=student.review_mode,
                    consecutive_errors=student.consecutive_errors.get(kc_id, 0),
                    consecutive_correct=student.consecutive_correct.get(kc_id, 0),
                    recent_accuracy=1.0 if payload["score"] >= 0.5 else 0.0,
                    irt_difficulty=payload.get("irt_difficulty"),
                )
                student.mastery[kc_id] = updated
                if payload["score"] >= 0.5:
                    student.consecutive_errors[kc_id] = 0
                    student.consecutive_correct[kc_id] = student.consecutive_correct.get(kc_id, 0) + 1
                else:
                    student.consecutive_errors[kc_id] = student.consecutive_errors.get(kc_id, 0) + 1
                    student.consecutive_correct[kc_id] = 0
                updated_mastery[kc_id] = updated
                consecutive_errors[kc_id] = student.consecutive_errors[kc_id]
            state["seen_task_ids"].add(str(payload["task_id"]))
            return {
                "updated_mastery": updated_mastery,
                "mastery_before": mastery_before,
                "consecutive_errors": consecutive_errors,
            }

        async def get_recommendation(_http, student_id, last_subject):
            remaining = [t for t in state["tasks"] if t["task_id"] not in state["seen_task_ids"]]
            if not remaining:
                raise httpx.HTTPStatusError(
                    "no tasks",
                    request=httpx.Request("POST", "http://retrieval/recommend"),
                    response=httpx.Response(404, json={"detail": "Нет доступных заданий"}),
                )
            return remaining[0]

        async def update_bandit_reward(_http, student_id, task_id, reward, **kwargs):
            state["reward_updates"].append({
                "student_id": str(student_id),
                "task_id": str(task_id),
                "reward": reward,
                **kwargs,
            })

        monkeypatch.setattr(main.clients, "submit_interaction", submit_interaction)
        monkeypatch.setattr(main.clients, "get_recommendation", get_recommendation)
        monkeypatch.setattr(main.clients, "update_bandit_reward", update_bandit_reward)
        monkeypatch.setattr(main, "assign_cluster_for_new_student", lambda *args, **kwargs: None)
        monkeypatch.setattr(main, "save_student_cluster", AsyncMock())

        yield ac, state


@pytest.mark.asyncio
async def test_full_session_flow_propagates_learning_signals(client):
    http, state = client
    student_id = "00000000-0000-0000-0000-000000000001"

    first = await http.get(f"/students/{student_id}/next-task")
    assert first.status_code == 200
    first_task = first.json()

    answer = await http.post(
        f"/sessions/{student_id}/answer",
        json={
            "task_id": first_task["task_id"],
            "part_id": "p1",
            "score": 1.0,
            "hints_used": 1,
            "time_spent_seconds": 37,
            "primary_kcs": ["kc_linear_eq_1var"],
            "secondary_kcs": [],
            "irt_difficulty": first_task["task"]["parts"][0]["irt_difficulty"],
            "recommendation_source": first_task["recommendation_source"],
            "last_subject": first_task["subject"],
        },
    )
    assert answer.status_code == 200
    body = answer.json()
    assert body["mastery_update"]["updated_mastery"]["kc_linear_eq_1var"] > 0.0
    assert body["next_task"]["task_id"] != first_task["task_id"]

    reward_update = state["reward_updates"][0]
    assert reward_update["raw_score"] == 1.0
    assert reward_update["hints_used"] == 1
    assert reward_update["time_spent_seconds"] == 37
    assert reward_update["irt_difficulty"] == first_task["task"]["parts"][0]["irt_difficulty"]
    assert reward_update["mastery_delta"] > 0.0


@pytest.mark.asyncio
async def test_full_session_flow_surfaces_task_exhaustion(client):
    http, _state = client
    student_id = "00000000-0000-0000-0000-000000000001"

    for _ in range(2):
        rec = await http.get(f"/students/{student_id}/next-task")
        task = rec.json()
        resp = await http.post(
            f"/sessions/{student_id}/answer",
            json={
                "task_id": task["task_id"],
                "part_id": "p1",
                "score": 0.0,
                "primary_kcs": ["kc_linear_eq_1var"],
                "secondary_kcs": [],
                "irt_difficulty": task["task"]["parts"][0]["irt_difficulty"],
                "recommendation_source": task["recommendation_source"],
                "last_subject": task["subject"],
            },
        )
        assert resp.status_code in (200, 404)

    final = await http.get(f"/students/{student_id}/next-task")
    assert final.status_code == 404
    assert "Нет доступных заданий" in final.text
