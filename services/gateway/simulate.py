"""
/simulate/* — HTTP-обёртки над Kafka для ручного тестирования.

Открой http://localhost:8005/docs чтобы увидеть Swagger UI.

Каждый эндпоинт имитирует то, что backend отправляет через Kafka,
и возвращает ответ Learnity синхронно (ждёт Kafka-ответа).
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from aiokafka import AIOKafkaConsumer
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import kafka_producer

KAFKA_BOOTSTRAP = kafka_producer.KAFKA_BOOTSTRAP

router = APIRouter(prefix="/simulate", tags=["Simulation (Kafka integration test)"])


# ─── Схемы ────────────────────────────────────────────────────────────────────

class StudentRegisteredRequest(BaseModel):
    student_id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, description="UUID студента (генерируется автоматически)")
    grade: int = Field(8, ge=5, le=11, description="Класс (5–11)")


class TaskRequestBody(BaseModel):
    student_id: uuid.UUID = Field(..., description="UUID студента")
    subject: str = Field("algebra", description="Предмет (algebra, geometry, ...)")


class TaskAnsweredRequest(BaseModel):
    student_id: uuid.UUID
    task_id: uuid.UUID
    score: float = Field(..., ge=0.0, le=1.0, description="0.0 = неверно, 1.0 = верно")
    topic: str = Field(..., description="kc_id темы (например kc_linear_eq_1var)")
    hints_used: int = 0
    difficulty: str = Field("3", description="Сложность '1'–'5'")
    subject: str = "algebra"
    recommendation_source: Optional[str] = None


class MasteryRequestBody(BaseModel):
    student_id: uuid.UUID
    skill_keys: list[str] = Field(default_factory=list, description="Список kc_id. Пустой = все KC")


class PlanRequestBody(BaseModel):
    student_id: uuid.UUID
    mode: str = Field("target_mastery", description="target_mastery | coverage")
    target_skill_key: Optional[str] = Field(None, description="Целевая тема (только для target_mastery)")
    mastery_threshold: float = Field(0.80, ge=0.0, le=1.0)


# ─── Вспомогательное: ждать ответ из Kafka ────────────────────────────────────

async def _wait_kafka_response(
    topic: str,
    match_key: str,
    match_value: str,
    timeout: float = 20.0,
) -> dict:
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=f"simulate-{uuid.uuid4()}",
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest",
    )
    await consumer.start()
    try:
        async def _read():
            async for msg in consumer:
                if str(msg.value.get(match_key)) == str(match_value):
                    return msg.value
        return await asyncio.wait_for(_read(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Learnity не ответил за {timeout:.0f}с (topic={topic})")
    finally:
        await consumer.stop()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Эндпоинты ────────────────────────────────────────────────────────────────

@router.post("/student-registered", summary="Зарегистрировать студента (→ Kafka)")
async def simulate_student_registered(req: StudentRegisteredRequest):
    """
    Публикует STUDENT_REGISTERED в `learnity.student.registered`.
    Learnity инициализирует профиль + cold-start mastery.

    Возвращает подтверждение (fire-and-forget, ответа в Kafka нет).
    """
    await kafka_producer.send("learnity.student.registered", {
        "event_type": "STUDENT_REGISTERED",
        "student_id": str(req.student_id),
        "grade": req.grade,
    })
    return {"published": True, "student_id": str(req.student_id), "grade": req.grade}


@router.post("/task-request", summary="Запросить задание (→ Kafka, ← ждёт TASK_RECOMMENDED)")
async def simulate_task_request(req: TaskRequestBody):
    """
    Публикует TASK_REQUESTED в `learnity.task.request` и ждёт TASK_RECOMMENDED.

    Learnity выбирает задание через Thompson Sampling + ZPD и отвечает через Kafka.
    """
    request_id = str(uuid.uuid4())

    wait_task = asyncio.create_task(
        _wait_kafka_response("learnity.task.recommended", "request_id", request_id)
    )

    await kafka_producer.send("learnity.task.request", {
        "event_type": "TASK_REQUESTED",
        "request_id": request_id,
        "student_id": str(req.student_id),
        "subject": req.subject,
    })

    return await wait_task


@router.post("/task-answered", summary="Ответить на задание (→ Kafka, fire-and-forget)")
async def simulate_task_answered(req: TaskAnsweredRequest):
    """
    Публикует TASK_ANSWERED в `learnity.task.answered`.
    Learnity обновляет mastery и модель бандита.
    """
    await kafka_producer.send("learnity.task.answered", {
        "event_type": "TASK_ANSWERED",
        "student_id": str(req.student_id),
        "task_id": str(req.task_id),
        "score": req.score,
        "hints_used": req.hints_used,
        "topic": req.topic,
        "subject": req.subject,
        "difficulty": req.difficulty,
        "recommendation_source": req.recommendation_source,
    })
    return {"published": True}


@router.post("/mastery-request", summary="Запросить mastery (→ Kafka, ← ждёт MASTERY_RESPONSE)")
async def simulate_mastery_request(req: MasteryRequestBody):
    """
    Публикует MASTERY_REQUESTED и ждёт MASTERY_RESPONSE.
    Если skill_keys пустой — возвращает mastery по всем KC.
    """
    request_id = str(uuid.uuid4())

    wait_task = asyncio.create_task(
        _wait_kafka_response("learnity.mastery.response", "request_id", request_id)
    )

    await kafka_producer.send("learnity.mastery.request", {
        "event_type": "MASTERY_REQUESTED",
        "request_id": request_id,
        "student_id": str(req.student_id),
        "skill_keys": req.skill_keys,
    })

    return await wait_task


@router.post("/plan-request", summary="Создать учебный план (→ Kafka, ← ждёт PLAN_CREATED)")
async def simulate_plan_request(req: PlanRequestBody):
    """
    Публикует PLAN_REQUESTED и ждёт PLAN_CREATED.
    """
    request_id = str(uuid.uuid4())

    wait_task = asyncio.create_task(
        _wait_kafka_response("learnity.plan.created", "request_id", request_id)
    )

    await kafka_producer.send("learnity.plan.request", {
        "event_type": "PLAN_REQUESTED",
        "request_id": request_id,
        "student_id": str(req.student_id),
        "mode": req.mode,
        "target_skill_key": req.target_skill_key,
        "mastery_threshold": req.mastery_threshold,
    })

    return await wait_task
