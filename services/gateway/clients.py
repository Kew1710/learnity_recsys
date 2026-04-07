"""HTTP-клиенты для обращения к Profile и Retrieval."""

import os
import uuid
from typing import Any

import httpx

PROFILE_URL   = os.getenv("PROFILE_URL",   "http://127.0.0.1:8001")
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://127.0.0.1:8004")
GRAPH_URL     = os.getenv("GRAPH_URL",     "http://127.0.0.1:8002")
MACRO_URL     = os.getenv("MACRO_URL",     "http://127.0.0.1:8006")
# Gateway itself runs on 8005 (8000 may be taken by other local projects)

TIMEOUT = 5.0


async def submit_interaction(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
    payload: dict,
) -> dict:
    resp = await client.post(
        f"{PROFILE_URL}/students/{student_id}/interactions",
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def assign_experiment(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
) -> dict:
    resp = await client.post(
        f"{PROFILE_URL}/students/{student_id}/experiment",
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def check_plan_thresholds(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
    updated_mastery: dict,
    consecutive_errors: dict,
) -> None:
    resp = await client.post(
        f"{PROFILE_URL}/students/{student_id}/plan/check",
        json={"updated_mastery": updated_mastery, "consecutive_errors": consecutive_errors},
        timeout=TIMEOUT,
    )
    # 404 значит нет плана — это нормально
    if resp.status_code not in (200, 404):
        resp.raise_for_status()


async def update_bandit_reward(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
    task_id: uuid.UUID,
    reward: float,
) -> None:
    resp = await client.patch(
        f"{RETRIEVAL_URL}/bandit_log/reward",
        json={"student_id": str(student_id), "task_id": str(task_id), "reward": reward},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()


async def get_recommendation(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
    last_subject: str | None,
) -> dict:
    body: dict[str, Any] = {"student_id": str(student_id)}
    if last_subject:
        body["last_subject"] = last_subject
    resp = await client.post(
        f"{RETRIEVAL_URL}/recommend",
        json=body,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def get_all_kcs(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(f"{GRAPH_URL}/kcs", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


async def cold_start_student(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
    kcs: list[dict],
) -> dict:
    resp = await client.post(
        f"{PROFILE_URL}/students/{student_id}/mastery/cold_start",
        json={"kcs": kcs},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def get_all_mastery(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
) -> list[dict]:
    resp = await client.get(
        f"{PROFILE_URL}/students/{student_id}/mastery",
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def create_plan(
    client: httpx.AsyncClient,
    student_id: uuid.UUID,
    mode: str,
    target_kc_id: str | None,
    mastery_threshold: float,
) -> dict:
    body: dict[str, Any] = {
        "student_id": str(student_id),
        "mode": mode,
        "mastery_threshold": mastery_threshold,
    }
    if target_kc_id:
        body["target_kc_id"] = target_kc_id
    resp = await client.post(f"{MACRO_URL}/plans", json=body, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


async def create_student(
    client: httpx.AsyncClient,
    grade: int,
    student_id: uuid.UUID | None = None,
    review_mode: bool = False,
) -> dict:
    body: dict[str, Any] = {"grade": grade, "review_mode": review_mode}
    if student_id:
        body["student_id"] = str(student_id)
    resp = await client.post(
        f"{PROFILE_URL}/students",
        json=body,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()
