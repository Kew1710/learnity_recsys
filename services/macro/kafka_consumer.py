"""Kafka consumer для получения MicroSummary событий от retrieval-сервиса."""

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import httpx
import sqlalchemy as sa
from aiokafka import AIOKafkaConsumer

from shared.db import AsyncSessionLocal
from . import clients, kafka_producer as macro_kafka
from .plan_lifecycle import (
    apply_plan_actions,
    check_and_advance,
    create_plateau_alert,
    evaluate_micro_summary,
)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_MICRO_SUMMARIES = "micro_summaries"

# Порог mastery: prereq считается «слабым» если ниже этого значения
WEAK_PREREQ_THRESHOLD = 0.60
# Минимальная сила связи: учитываем только сильные prereqs
STRONG_PREREQ_MIN_STRENGTH = 0.7

logger = logging.getLogger(__name__)


async def _get_active_plan_step(student_id: uuid.UUID) -> tuple[uuid.UUID, str, int | None] | None:
    """Возвращает (plan_id, kc_id, tasks_budget) активного шага последнего плана."""
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            sa.text("""
                SELECT ps.plan_id, ps.kc_id, ps.tasks_budget
                FROM plan_steps ps
                JOIN learning_plans lp ON lp.id = ps.plan_id
                WHERE lp.student_id = :student_id
                  AND ps.status = 'in_progress'
                  AND lp.id = (
                      SELECT id FROM learning_plans
                      WHERE student_id = :student_id
                      ORDER BY created_at DESC LIMIT 1
                  )
                LIMIT 1
            """),
            {"student_id": student_id},
        )).fetchone()
    if not row:
        return None
    return row[0], row[1], row[2]


async def _create_replan_alert(
    student_id: uuid.UUID,
    plan_id: uuid.UUID,
    kc_id: str,
    mastery_current: float,
    tasks_spent: int,
    reason: str,
) -> None:
    """Сохраняет сигнал replan для последующей ручной обработки."""
    async with AsyncSessionLocal() as db:
        await db.execute(
            sa.text("""
                INSERT INTO teacher_alerts
                    (id, student_id, plan_id, kc_id, alert_type,
                     mastery_at_alert, tasks_spent, message, created_at)
                VALUES
                    (:id, :student_id, :plan_id, :kc_id, 'replan_requested',
                     :mastery, :tasks_spent, :message, :now)
            """),
            {
                "id": uuid.uuid4(),
                "student_id": student_id,
                "plan_id": plan_id,
                "kc_id": kc_id,
                "mastery": mastery_current,
                "tasks_spent": tasks_spent,
                "message": reason,
                "now": datetime.now(timezone.utc),
            },
        )
        await db.commit()

    try:
        await macro_kafka.send_teacher_alert(
            student_id=str(student_id),
            alert_type="replan_requested",
            skill_key=kc_id,
            mastery_at_alert=mastery_current,
            tasks_spent=tasks_spent,
            message=reason,
        )
    except Exception:
        logger.warning("Failed to publish replan alert for student=%s kc=%s", student_id, kc_id)


async def _find_weakest_prereq(student_id: uuid.UUID, kc_id: str) -> str | None:
    """
    Находит слабейший пре-реквизит KC для данного студента.

    Алгоритм:
      1. Получаем список пре-реквизитов KC из graph-сервиса
      2. Фильтруем по силе связи (>= STRONG_PREREQ_MIN_STRENGTH)
      3. Берём mastery студента для каждого prereq
      4. Возвращаем kc_id с минимальным mastery, если оно ниже WEAK_PREREQ_THRESHOLD

    Returns None если слабых prereqs нет или запросы не удались.
    """
    async with httpx.AsyncClient(trust_env=False) as http:
        try:
            prereqs = await clients.get_all_prerequisites(http, kc_id)
            mastery = await clients.get_student_mastery(http, student_id)
        except Exception:
            return None

    strong = [p for p in prereqs if float(p.get("strength", 0)) >= STRONG_PREREQ_MIN_STRENGTH]
    if not strong:
        return None

    weakest_id = min(strong, key=lambda p: mastery.get(p["kc_id"], 0.0))["kc_id"]
    weakest_mastery = mastery.get(weakest_id, 0.0)

    if weakest_mastery < WEAK_PREREQ_THRESHOLD:
        return weakest_id

    return None


async def _handle_summary(summary: dict) -> None:
    student_id_raw = summary.get("student_id")
    kc_id = summary.get("kc_id")
    mastery_current = summary.get("mastery_current", 0.0)

    if not student_id_raw or not kc_id:
        logger.warning("MicroSummary missing student_id or kc_id, skipping")
        return

    try:
        student_id = uuid.UUID(student_id_raw)
    except ValueError:
        logger.warning("Invalid student_id in MicroSummary: %s", student_id_raw)
        return

    tasks_spent = summary.get("tasks_spent", 0)
    velocity = summary.get("velocity", 0.0)
    recent_accuracy = summary.get("avg_score", 1.0)

    # 1. Автопереход шага если mastery достиг порога И recent_accuracy достаточна
    advanced = await check_and_advance(student_id, kc_id, mastery_current, recent_accuracy)
    if advanced:
        logger.info(
            "Plan step auto-advanced: student=%s kc=%s mastery=%.3f",
            student_id, kc_id, mastery_current,
        )
        return

    active_step = await _get_active_plan_step(student_id)
    if not active_step:
        return
    plan_id, active_kc_id, tasks_budget = active_step
    if active_kc_id != kc_id:
        logger.info(
            "MicroSummary kc=%s не совпадает с активным шагом kc=%s для student=%s",
            kc_id,
            active_kc_id,
            student_id,
        )
        return

    weakest_prereq_kc_id = await _find_weakest_prereq(student_id, kc_id)
    actions = evaluate_micro_summary(
        summary,
        weakest_prereq_kc_id=weakest_prereq_kc_id,
        tasks_budget=tasks_budget,
    )
    if actions:
        non_replan_actions = [a for a in actions if a.action_type != "replan"]
        if non_replan_actions:
            await apply_plan_actions(student_id, non_replan_actions)
            logger.info(
                "Applied lifecycle actions: student=%s kc=%s actions=%s",
                student_id,
                kc_id,
                [a.action_type for a in non_replan_actions],
            )
        for action in actions:
            if action.action_type == "replan":
                reason = action.payload.get("reason", "replan requested by micro-summary")
                await _create_replan_alert(
                    student_id=student_id,
                    plan_id=plan_id,
                    kc_id=kc_id,
                    mastery_current=mastery_current,
                    tasks_spent=tasks_spent,
                    reason=reason,
                )
                logger.info(
                    "Replan requested: student=%s kc=%s reason=%s",
                    student_id,
                    kc_id,
                    reason,
                )

    # 2. Plateau → teacher_alert если velocity≈0 за последние 20 задач
    #    Порог: tasks_spent >= 50 чтобы не спамить в начале шага
    PLATEAU_MIN_TASKS = 50
    if tasks_spent >= PLATEAU_MIN_TASKS and abs(velocity) < 0.01:
        await create_plateau_alert(student_id, plan_id, kc_id, mastery_current, tasks_spent)
        logger.info(
            "Plateau alert created: student=%s kc=%s velocity=%.3f tasks_spent=%d",
            student_id, kc_id, velocity, tasks_spent,
        )



async def run_consumer() -> None:
    consumer = AIOKafkaConsumer(
        TOPIC_MICRO_SUMMARIES,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="macro-planner",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    await consumer.start()
    try:
        async for msg in consumer:
            summary = msg.value
            logger.info(
                "MicroSummary received: student=%s kc=%s velocity=%.3f tasks_spent=%d",
                summary.get("student_id"),
                summary.get("kc_id"),
                summary.get("velocity", 0.0),
                summary.get("tasks_spent", 0),
            )
            try:
                await _handle_summary(summary)
            except Exception:
                logger.exception("Error handling MicroSummary: %s", summary)
    finally:
        await consumer.stop()
