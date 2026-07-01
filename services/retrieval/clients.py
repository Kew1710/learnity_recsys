"""HTTP-клиенты для обращения к другим сервисам."""

import os
import uuid
from typing import Any

import httpx

PROFILE_URL = os.getenv("PROFILE_URL", "http://127.0.0.1:8001")
GRAPH_URL = os.getenv("GRAPH_URL", "http://127.0.0.1:8002")
TASK_BANK_URL = os.getenv("TASK_BANK_URL", "http://127.0.0.1:8003")

TIMEOUT = 5.0


async def get_student(client: httpx.AsyncClient, student_id: uuid.UUID) -> dict:
    resp = await client.get(f"{PROFILE_URL}/students/{student_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


async def apply_diagnostic_mastery(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
    mastery: dict[str, float],
) -> dict:
    resp = await client.post(
        f"{PROFILE_URL}/students/{student_id}/mastery/diagnostic",
        json={"mastery": mastery},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def get_seen_tasks(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
    ttl_days: int | None = None,
) -> list[str]:
    params = {"ttl_days": ttl_days} if ttl_days is not None else {}
    resp = await client.get(
        f"{PROFILE_URL}/students/{student_id}/seen_tasks",
        params=params,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["task_ids"]


async def get_kc_prerequisites(client: httpx.AsyncClient, kc_id: str) -> list[dict]:
    """Возвращает [{kc_id, strength}] пре-реквизитов для KC, отсортированных по убыванию strength."""
    resp = await client.get(f"{GRAPH_URL}/nodes/{kc_id}/prerequisites", timeout=TIMEOUT)
    resp.raise_for_status()
    prereqs = resp.json()
    return sorted(prereqs, key=lambda p: p["strength"], reverse=True)


async def get_zpd(
    client: httpx.AsyncClient,
    mastery: dict[str, float],
    student_grade: int,
) -> list[dict]:
    resp = await client.post(
        f"{GRAPH_URL}/zpd",
        json={"mastery": mastery, "student_grade": student_grade},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def get_tasks_by_kc(
    client: httpx.AsyncClient,
    kc_id: str,
    grade_min: int,
    exclude_task_ids: list[str],
    limit: int = 10,
) -> list[dict]:
    resp = await client.get(
        f"{TASK_BANK_URL}/tasks/by_kc",
        params={
            "kc_id": kc_id,
            "grade_min": grade_min,
            "exclude": exclude_task_ids,
            "limit": limit,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()
