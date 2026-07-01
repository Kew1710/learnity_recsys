"""API Gateway — единая точка входа для клиента.

Горячий путь (на каждый ответ ученика):
  POST /sessions/{student_id}/answer
    1. Отправить результат в Profile (обновить mastery)
    2. Запросить следующее задание из Retrieval
    3. Вернуть задание клиенту

Вспомогательные:
  POST /students         — создать нового ученика
  GET  /students/{id}/next-task — получить следующее задание без submit
  GET  /health
"""

import asyncio
import os
import uuid
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shared.config import gateway as _cfg
from . import clients, kafka_producer
from .kafka_consumer import run_consumer
from .simulate import router as simulate_router
from services.clustering.cluster import assign_cluster_for_new_student, save_student_cluster, ensure_centroids_loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    await kafka_producer.start()
    await ensure_centroids_loaded()
    task = asyncio.create_task(run_consumer())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await kafka_producer.stop()


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

app = FastAPI(title="Learnity Gateway", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(simulate_router)
app.mount("/static", StaticFiles(directory="services/gateway/static"), name="static")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SubmitAnswerRequest(BaseModel):
    task_id: uuid.UUID
    part_id: str
    score: float                        # 0.0–1.0
    hints_used: int = 0
    time_spent_seconds: int = 0
    misconception_triggered: Optional[str] = None

    primary_kcs: list[str]
    secondary_kcs: list[str] = []

    p_transit: float = 0.1
    p_slip: float = 0.1
    p_guess: float = 0.2
    irt_difficulty: Optional[float] = None
    half_life_days: float = 45.0
    recommendation_source: Optional[str] = None

    # Для subject rotation в Retrieval
    last_subject: Optional[str] = None


class CreateStudentRequest(BaseModel):
    grade: int
    student_id: Optional[uuid.UUID] = None
    review_mode: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/students", status_code=201)
async def create_student(req: CreateStudentRequest):
    async with httpx.AsyncClient(trust_env=False) as http:
        try:
            result = await clients.create_student(
                http,
                grade=req.grade,
                student_id=req.student_id,
                review_mode=req.review_mode,
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail="Profile service error") from e

        student_id = result["student_id"]

        # Cold-start: выставить mastery на основе класса ученика
        cold_start_mastery: dict[str, float] = {}
        try:
            kcs = await clients.get_all_kcs(http)
            cold_start_result = await clients.cold_start_student(http, student_id, kcs)
            cold_start_mastery = cold_start_result.get("mastery", {})
        except httpx.HTTPStatusError as e:
            # cold_start не критичен — ученик создан, продолжаем без инициализации mastery
            logger.warning("cold_start skipped for student=%s: %s", student_id, e)

        # Назначение в A/B эксперимент (50/50)
        try:
            await clients.assign_experiment(http, student_id)
        except httpx.HTTPStatusError as e:
            logger.warning("experiment assignment failed for student=%s: %s", student_id, e)

        # Назначение кластера по ближайшему центроиду
        try:
            cluster_id = assign_cluster_for_new_student(cold_start_mastery)
            if cluster_id is not None:
                await save_student_cluster(
                    uuid.UUID(student_id),
                    cluster_id,
                    reason="cold_start",
                    task_count=0,
                )
        except Exception as e:
            logger.warning("cluster assignment failed for student=%s: %s", student_id, e)

    return result


@app.post("/sessions/{student_id}/answer")
async def submit_answer(student_id: uuid.UUID, req: SubmitAnswerRequest):
    """
    Главный эндпоинт горячего пути.
    Принимает ответ ученика, обновляет mastery, возвращает следующее задание.
    """
    started = time.perf_counter()
    async with httpx.AsyncClient(trust_env=False) as http:

        # 1. Обновить mastery в Profile
        interaction_payload = {
            "task_id": str(req.task_id),
            "part_id": req.part_id,
            "score": req.score,
            "hints_used": req.hints_used,
            "time_spent_seconds": req.time_spent_seconds,
            "misconception_triggered": req.misconception_triggered,
            "primary_kcs": req.primary_kcs,
            "secondary_kcs": req.secondary_kcs,
            "p_transit": req.p_transit,
            "p_slip": req.p_slip,
            "p_guess": req.p_guess,
            "irt_difficulty": req.irt_difficulty,
            "half_life_days": req.half_life_days,
            "recommendation_source": req.recommendation_source,
        }
        try:
            mastery_update = await clients.submit_interaction(http, student_id, interaction_payload)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Student not found")
            raise HTTPException(status_code=502, detail="Profile service error") from e

        # 2. Обновить reward в bandit_log (не критично, не блокируем основной путь)
        primary_kcs = req.primary_kcs
        if primary_kcs:
            before = mastery_update.get("mastery_before", {})
            after = mastery_update.get("updated_mastery", {})
            cons_errors = mastery_update.get("consecutive_errors", {})

            deltas = [after.get(kc, 0.0) - before.get(kc, 0.0) for kc in primary_kcs]
            delta_mastery = sum(deltas) / len(deltas)

            # Штраф за фрустрацию: 3+ ошибок подряд
            frustration = 1.0 if any(cons_errors.get(kc, 0) >= 3 for kc in primary_kcs) else 0.0
            # Штраф за скуку: уже освоенная тема (mastery > 0.9) решена правильно
            boredom = 1.0 if (
                req.score >= 0.5
                and any(before.get(kc, 0.0) > 0.9 for kc in primary_kcs)
            ) else 0.0

            reward = delta_mastery - _cfg.REWARD_BETA * frustration - _cfg.REWARD_GAMMA * boredom

            try:
                await clients.update_bandit_reward(
                    http,
                    student_id,
                    req.task_id,
                    reward,
                    raw_score=req.score,
                    hints_used=req.hints_used,
                    time_spent_seconds=req.time_spent_seconds,
                    irt_difficulty=req.irt_difficulty,
                    mastery_delta=delta_mastery,
                )
            except httpx.HTTPStatusError as e:
                logger.warning("bandit reward update failed for student=%s task=%s: %s", student_id, req.task_id, e)

        # 3. Plan thresholds — handled by Macro via MicroSummary/Kafka (see П2.3)

        # 4. Получить следующее задание
        try:
            recommendation = await clients.get_recommendation(http, student_id, req.last_subject)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail=e.response.json().get("detail", "No tasks available"))
            raise HTTPException(status_code=502, detail="Retrieval service error") from e

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "submit_answer completed student=%s task=%s score=%.3f source=%s latency_ms=%.1f",
        student_id,
        req.task_id,
        req.score,
        req.recommendation_source,
        elapsed_ms,
    )

    return {
        "mastery_update": mastery_update,
        "next_task": recommendation,
    }


@app.get("/students/{student_id}/next-task")
async def get_next_task(student_id: uuid.UUID, last_subject: Optional[str] = None):
    """Получить следующее задание без submit (например, при первом входе)."""
    async with httpx.AsyncClient(trust_env=False) as http:
        try:
            recommendation = await clients.get_recommendation(http, student_id, last_subject)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail=e.response.json().get("detail", "No tasks available"))
            raise HTTPException(status_code=502, detail="Retrieval service error") from e
    return recommendation


# ---------------------------------------------------------------------------
# Proxy → backend (чтобы integration-test.html не получал CORS)
# ---------------------------------------------------------------------------

@app.api_route("/backend/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_backend(path: str, request: Request):
    url = f"{BACKEND_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length", "origin", "referer")}
    async with httpx.AsyncClient(trust_env=False, timeout=30) as http:
        resp = await http.request(method=request.method, url=url, content=body, headers=headers)
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
