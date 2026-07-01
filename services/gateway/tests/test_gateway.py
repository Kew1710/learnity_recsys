"""Юнит-тесты Gateway. Все внешние сервисы замоканы через respx."""

import json
import uuid
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from services.gateway import main
from services.gateway.main import app

STUDENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
PROFILE_BASE = "http://127.0.0.1:8001"
RETRIEVAL_BASE = "http://127.0.0.1:8004"
GRAPH_BASE = "http://127.0.0.1:8002"

TASK_STUB = {
    "task_id": "task-001",
    "kc_id": "kc_linear_eq_1var",
    "subject": "algebra",
    "recommendation_source": "zpd",
    "half_life_days": 45.0,
    "task": {
        "task_id": "task-001",
        "parts": [{"irt_difficulty": 0.5, "primary_kcs": ["kc_linear_eq_1var"]}],
    },
}

MASTERY_STUB = {
    "updated_mastery": {"kc_linear_eq_1var": 0.72},
    "mastery_before": {"kc_linear_eq_1var": 0.60},
}


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def stub_cluster_hooks(monkeypatch):
    monkeypatch.setattr(main, "assign_cluster_for_new_student", lambda mastery: None)
    monkeypatch.setattr(main, "save_student_cluster", AsyncMock())


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /students
# ---------------------------------------------------------------------------

KCS_STUB = [
    {"kc_id": "kc_arithmetic_basic", "grade_introduced": 5},
    {"kc_id": "kc_linear_eq_1var", "grade_introduced": 7},
]


@respx.mock
async def test_create_student(client):
    respx.post(f"{PROFILE_BASE}/students").mock(
        return_value=httpx.Response(201, json={"student_id": str(STUDENT_ID), "grade": 7, "review_mode": False})
    )
    respx.get(f"{GRAPH_BASE}/kcs").mock(
        return_value=httpx.Response(200, json=KCS_STUB)
    )
    respx.post(f"{PROFILE_BASE}/students/{STUDENT_ID}/mastery/cold_start").mock(
        return_value=httpx.Response(200, json={"initialized_count": 2, "mastery": {"kc_arithmetic_basic": 1.0}})
    )
    respx.post(f"{PROFILE_BASE}/students/{STUDENT_ID}/experiment").mock(
        return_value=httpx.Response(201, json={"experiment_id": "linucb_v1", "variant": "treatment"})
    )
    resp = await client.post("/students", json={"grade": 7})
    assert resp.status_code == 201
    assert resp.json()["grade"] == 7


@respx.mock
async def test_create_student_profile_down(client):
    respx.post(f"{PROFILE_BASE}/students").mock(
        return_value=httpx.Response(500)
    )
    resp = await client.post("/students", json={"grade": 7})
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /sessions/{student_id}/answer
# ---------------------------------------------------------------------------

def _answer_payload(**overrides):
    base = {
        "task_id": str(uuid.uuid4()),
        "part_id": "p1",
        "score": 1.0,
        "primary_kcs": ["kc_linear_eq_1var"],
        "last_subject": "algebra",
    }
    base.update(overrides)
    return base


def _mock_plan_check():
    respx.post(f"{PROFILE_BASE}/students/{STUDENT_ID}/plan/check").mock(
        return_value=httpx.Response(200, json={"changes": []})
    )


@respx.mock
async def test_submit_answer_happy_path(client):
    reward_route = respx.patch(f"{RETRIEVAL_BASE}/bandit_log/reward").mock(
        return_value=httpx.Response(200, json={"updated": True})
    )
    respx.post(f"{PROFILE_BASE}/students/{STUDENT_ID}/interactions").mock(
        return_value=httpx.Response(201, json=MASTERY_STUB)
    )
    _mock_plan_check()
    respx.post(f"{RETRIEVAL_BASE}/recommend").mock(
        return_value=httpx.Response(200, json=TASK_STUB)
    )
    resp = await client.post(f"/sessions/{STUDENT_ID}/answer", json=_answer_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert "mastery_update" in body
    assert "next_task" in body
    assert body["next_task"]["task_id"] == "task-001"
    reward_payload = json.loads(reward_route.calls[0].request.content)
    assert reward_payload["raw_score"] == 1.0
    assert reward_payload["hints_used"] == 0
    assert reward_payload["time_spent_seconds"] == 0
    assert reward_payload["irt_difficulty"] is None
    assert reward_payload["mastery_delta"] == pytest.approx(0.12)


@respx.mock
async def test_submit_answer_student_not_found(client):
    respx.post(f"{PROFILE_BASE}/students/{STUDENT_ID}/interactions").mock(
        return_value=httpx.Response(404, json={"detail": "Student not found"})
    )
    resp = await client.post(f"/sessions/{STUDENT_ID}/answer", json=_answer_payload())
    assert resp.status_code == 404


@respx.mock
async def test_submit_answer_no_tasks(client):
    respx.post(f"{PROFILE_BASE}/students/{STUDENT_ID}/interactions").mock(
        return_value=httpx.Response(201, json=MASTERY_STUB)
    )
    respx.patch(f"{RETRIEVAL_BASE}/bandit_log/reward").mock(
        return_value=httpx.Response(200, json={"updated": True})
    )
    _mock_plan_check()
    respx.post(f"{RETRIEVAL_BASE}/recommend").mock(
        return_value=httpx.Response(404, json={"detail": "Нет доступных заданий"})
    )
    resp = await client.post(f"/sessions/{STUDENT_ID}/answer", json=_answer_payload())
    assert resp.status_code == 404


@respx.mock
async def test_submit_answer_retrieval_down(client):
    respx.post(f"{PROFILE_BASE}/students/{STUDENT_ID}/interactions").mock(
        return_value=httpx.Response(201, json=MASTERY_STUB)
    )
    respx.patch(f"{RETRIEVAL_BASE}/bandit_log/reward").mock(
        return_value=httpx.Response(200, json={"updated": True})
    )
    _mock_plan_check()
    respx.post(f"{RETRIEVAL_BASE}/recommend").mock(
        return_value=httpx.Response(503)
    )
    resp = await client.post(f"/sessions/{STUDENT_ID}/answer", json=_answer_payload())
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# GET /students/{id}/next-task
# ---------------------------------------------------------------------------

@respx.mock
async def test_get_next_task(client):
    respx.post(f"{RETRIEVAL_BASE}/recommend").mock(
        return_value=httpx.Response(200, json=TASK_STUB)
    )
    resp = await client.get(f"/students/{STUDENT_ID}/next-task")
    assert resp.status_code == 200
    assert resp.json()["subject"] == "algebra"


@respx.mock
async def test_get_next_task_with_last_subject(client):
    route = respx.post(f"{RETRIEVAL_BASE}/recommend").mock(
        return_value=httpx.Response(200, json=TASK_STUB)
    )
    resp = await client.get(f"/students/{STUDENT_ID}/next-task?last_subject=algebra")
    assert resp.status_code == 200
    # Убедимся что last_subject передан в тело запроса к Retrieval
    sent = route.calls[0].request
    body = json.loads(sent.content)
    assert body.get("last_subject") == "algebra"
