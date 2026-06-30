"""HTTP-клиенты macro-сервиса."""

import os
import uuid

import httpx

GRAPH_URL = os.getenv("GRAPH_URL", "http://127.0.0.1:8002")
PROFILE_URL = os.getenv("PROFILE_URL", "http://127.0.0.1:8001")
TASK_BANK_URL = os.getenv("TASK_BANK_URL", "http://127.0.0.1:8003")
TIMEOUT = 5.0


async def get_all_prerequisites(client: httpx.AsyncClient, kc_id: str) -> list[dict]:
    """Возвращает [{kc_id, strength}] пре-реквизитов для KC."""
    resp = await client.get(f"{GRAPH_URL}/nodes/{kc_id}/prerequisites", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


async def get_student_mastery(client: httpx.AsyncClient, student_id: uuid.UUID) -> dict[str, float]:
    """Возвращает {kc_id: mastery_effective} для ученика."""
    resp = await client.get(f"{PROFILE_URL}/students/{student_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return {
        kc_id: v["probability_effective"]
        for kc_id, v in data.get("mastery", {}).items()
    }


async def get_student_mastery_detailed(
    client: httpx.AsyncClient, student_id: uuid.UUID,
) -> dict[str, dict]:
    """Возвращает {kc_id: {probability_effective, confidence, attempts_count}} для ученика."""
    resp = await client.get(f"{PROFILE_URL}/students/{student_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return {
        kc_id: {
            "probability_effective": v["probability_effective"],
            "confidence": v.get("confidence", 0.0),
            "attempts_count": v.get("attempts_count", 0),
        }
        for kc_id, v in data.get("mastery", {}).items()
    }


async def get_student_grade(client: httpx.AsyncClient, student_id: uuid.UUID) -> int:
    resp = await client.get(f"{PROFILE_URL}/students/{student_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["grade"]


async def get_kc_grade(client: httpx.AsyncClient, kc_id: str) -> int | None:
    """Возвращает grade_introduced для KC из кэшированного списка графа, или None если не найдена."""
    resp = await client.get(f"{GRAPH_URL}/kcs", timeout=TIMEOUT)
    resp.raise_for_status()
    for kc in resp.json():
        if kc["kc_id"] == kc_id:
            return kc["grade_introduced"]
    return None


async def get_all_kc_grades(client: httpx.AsyncClient) -> dict[str, int]:
    """Возвращает {kc_id: grade_introduced} для всех KC из кэша графа."""
    resp = await client.get(f"{GRAPH_URL}/kcs", timeout=TIMEOUT)
    resp.raise_for_status()
    return {kc["kc_id"]: kc["grade_introduced"] for kc in resp.json()}


async def get_task_count_for_kc(client: httpx.AsyncClient, kc_id: str) -> int:
    resp = await client.get(f"{TASK_BANK_URL}/tasks/by_kc/count", params={"kc_id": kc_id}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("count", 0)


async def build_full_graph(
    client: httpx.AsyncClient,
    node_ids: list[str],
) -> dict[str, list[dict]]:
    """
    Строит полный граф пре-реквизитов для заданного набора KC.
    Returns {kc_id: [{kc_id: prereq_id, strength}]}.
    """
    graph: dict[str, list[dict]] = {}
    for kc_id in node_ids:
        try:
            prereqs = await get_all_prerequisites(client, kc_id)
            graph[kc_id] = prereqs
        except httpx.HTTPStatusError:
            graph[kc_id] = []
    return graph
