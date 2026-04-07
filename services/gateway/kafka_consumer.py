"""Kafka consumer — обработка входящих событий backend → Learnity."""

import logging
import os
import uuid

import httpx
from aiokafka import AIOKafkaConsumer

from services.clustering.cluster import assign_cluster_for_new_student, save_student_cluster
from . import clients, kafka_producer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

TOPIC_STUDENT_REGISTERED = "learnity.student.registered"
TOPIC_TASK_ANSWERED      = "learnity.task.answered"
TOPIC_TASK_REQUEST       = "learnity.task.request"
TOPIC_MASTERY_REQUEST    = "learnity.mastery.request"
TOPIC_PLAN_REQUEST       = "learnity.plan.request"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_student_registered(event: dict) -> None:
    student_id_raw = event.get("student_id")
    grade = event.get("grade")
    if not student_id_raw or grade is None:
        logger.warning("STUDENT_REGISTERED missing fields: %s", event)
        return

    student_id = uuid.UUID(student_id_raw)

    async with httpx.AsyncClient(trust_env=False) as http:
        try:
            await clients.create_student(http, grade=grade, student_id=student_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                logger.info("Student already exists: %s", student_id)
            else:
                logger.error("create_student failed for %s: %s", student_id, e)
                return

        cold_start_mastery: dict[str, float] = {}
        try:
            kcs = await clients.get_all_kcs(http)
            result = await clients.cold_start_student(http, student_id, kcs)
            cold_start_mastery = result.get("mastery", {})
        except httpx.HTTPStatusError as e:
            logger.warning("cold_start failed for %s: %s", student_id, e)

        try:
            await clients.assign_experiment(http, student_id)
        except httpx.HTTPStatusError as e:
            logger.warning("experiment assignment failed for %s: %s", student_id, e)

    try:
        cluster_id = assign_cluster_for_new_student(cold_start_mastery)
        if cluster_id is not None:
            await save_student_cluster(student_id, cluster_id)
    except Exception as e:
        logger.warning("cluster assignment failed for %s: %s", student_id, e)

    logger.info("STUDENT_REGISTERED handled: student=%s grade=%s", student_id, grade)


async def _handle_task_answered(event: dict) -> None:
    student_id_raw = event.get("student_id")
    task_id_raw = event.get("task_id")
    score = event.get("score")
    if not student_id_raw or not task_id_raw or score is None:
        logger.warning("TASK_ANSWERED missing fields: %s", event)
        return

    student_id = uuid.UUID(student_id_raw)
    task_id = uuid.UUID(task_id_raw)
    topic = event.get("topic", "")

    # Маппинг difficulty "1"–"5" → float 0.0–1.0
    difficulty_str = event.get("difficulty", "3")
    try:
        irt_difficulty = (int(difficulty_str) - 1) / 4.0
    except (ValueError, TypeError):
        irt_difficulty = None

    payload = {
        "task_id": str(task_id),
        "part_id": "main",
        "score": score,
        "hints_used": event.get("hints_used", 0),
        "time_spent_seconds": 0,
        "misconception_triggered": None,
        "primary_kcs": [topic] if topic else [],
        "secondary_kcs": [],
        "p_transit": 0.1,
        "p_slip": 0.1,
        "p_guess": 0.2,
        "irt_difficulty": irt_difficulty,
        "half_life_days": 45.0,
        "recommendation_source": event.get("recommendation_source"),
    }

    async with httpx.AsyncClient(trust_env=False) as http:
        try:
            mastery_update = await clients.submit_interaction(http, student_id, payload)
        except httpx.HTTPStatusError as e:
            logger.error("submit_interaction failed for student=%s: %s", student_id, e)
            return

        # Обновить reward бандита
        if topic:
            before = mastery_update.get("mastery_before", {})
            after = mastery_update.get("updated_mastery", {})
            delta = after.get(topic, 0.0) - before.get(topic, 0.0)
            cons_errors = mastery_update.get("consecutive_errors", {})
            frustration = 1.0 if cons_errors.get(topic, 0) >= 3 else 0.0
            boredom = 1.0 if score >= 0.5 and before.get(topic, 0.0) > 0.9 else 0.0
            reward = delta - 0.3 * frustration - 0.1 * boredom
            try:
                await clients.update_bandit_reward(http, student_id, task_id, reward)
            except httpx.HTTPStatusError as e:
                logger.warning("bandit reward failed for student=%s: %s", student_id, e)

        # Проверить пороги плана
        try:
            await clients.check_plan_thresholds(
                http, student_id,
                updated_mastery=mastery_update.get("updated_mastery", {}),
                consecutive_errors=mastery_update.get("consecutive_errors", {}),
            )
        except httpx.HTTPStatusError as e:
            logger.warning("plan threshold check failed for student=%s: %s", student_id, e)

    logger.info("TASK_ANSWERED handled: student=%s task=%s score=%.2f", student_id, task_id, score)


async def _handle_task_request(event: dict) -> None:
    request_id = event.get("request_id")
    student_id_raw = event.get("student_id")
    subject = event.get("subject")
    if not request_id or not student_id_raw:
        logger.warning("TASK_REQUESTED missing fields: %s", event)
        return

    student_id = uuid.UUID(student_id_raw)

    async with httpx.AsyncClient(trust_env=False) as http:
        try:
            rec = await clients.get_recommendation(http, student_id, last_subject=subject)
        except httpx.HTTPStatusError as e:
            logger.error("get_recommendation failed for student=%s: %s", student_id, e)
            return

    await kafka_producer.send(kafka_producer.TOPIC_TASK_RECOMMENDED, {
        "event_type": "TASK_RECOMMENDED",
        "request_id": request_id,
        "student_id": str(student_id),
        "task_id": rec["task_id"],
        "topic": rec["kc_id"],
        "recommendation_source": rec["recommendation_source"],
    })
    logger.info(
        "TASK_REQUESTED handled: student=%s task=%s source=%s",
        student_id, rec["task_id"], rec["recommendation_source"],
    )


async def _handle_mastery_request(event: dict) -> None:
    request_id = event.get("request_id")
    student_id_raw = event.get("student_id")
    skill_keys: list[str] = event.get("skill_keys") or []
    if not request_id or not student_id_raw:
        logger.warning("MASTERY_REQUESTED missing fields: %s", event)
        return

    student_id = uuid.UUID(student_id_raw)

    async with httpx.AsyncClient(trust_env=False) as http:
        try:
            records = await clients.get_all_mastery(http, student_id)
        except httpx.HTTPStatusError as e:
            logger.error("get_all_mastery failed for student=%s: %s", student_id, e)
            return

    # records — list[{kc_id, probability, probability_effective, ...}]
    mastery_map = {r["kc_id"]: r.get("probability_effective", r.get("probability", 0.0)) for r in records}

    if skill_keys:
        levels = [
            {"skill_key": k, "level": mastery_map.get(k, 0.0)}
            for k in skill_keys
        ]
    else:
        levels = [{"skill_key": k, "level": v} for k, v in mastery_map.items()]

    await kafka_producer.send(kafka_producer.TOPIC_MASTERY_RESPONSE, {
        "event_type": "MASTERY_RESPONSE",
        "request_id": request_id,
        "student_id": str(student_id),
        "levels": levels,
    })
    logger.info("MASTERY_REQUESTED handled: student=%s skills=%d", student_id, len(levels))


async def _handle_plan_request(event: dict) -> None:
    request_id = event.get("request_id")
    student_id_raw = event.get("student_id")
    mode = event.get("mode", "target_mastery")
    target_skill_key = event.get("target_skill_key")
    mastery_threshold = event.get("mastery_threshold", 0.80)
    if not request_id or not student_id_raw:
        logger.warning("PLAN_REQUESTED missing fields: %s", event)
        return

    student_id = uuid.UUID(student_id_raw)

    async with httpx.AsyncClient(trust_env=False) as http:
        try:
            result = await clients.create_plan(
                http,
                student_id=student_id,
                mode=mode,
                target_kc_id=target_skill_key,
                mastery_threshold=mastery_threshold,
            )
        except httpx.HTTPStatusError as e:
            logger.error("create_plan failed for student=%s: %s", student_id, e)
            return

    await kafka_producer.send(kafka_producer.TOPIC_PLAN_CREATED, {
        "event_type": "PLAN_CREATED",
        "request_id": request_id,
        "student_id": str(student_id),
        "plan_id": result["plan_id"],
        "steps_count": result["steps_count"],
    })
    logger.info(
        "PLAN_REQUESTED handled: student=%s plan=%s steps=%d",
        student_id, result["plan_id"], result["steps_count"],
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_HANDLERS = {
    "STUDENT_REGISTERED": _handle_student_registered,
    "TASK_ANSWERED":      _handle_task_answered,
    "TASK_REQUESTED":     _handle_task_request,
    "MASTERY_REQUESTED":  _handle_mastery_request,
    "PLAN_REQUESTED":     _handle_plan_request,
}


async def run_consumer() -> None:
    consumer = AIOKafkaConsumer(
        TOPIC_STUDENT_REGISTERED,
        TOPIC_TASK_ANSWERED,
        TOPIC_TASK_REQUEST,
        TOPIC_MASTERY_REQUEST,
        TOPIC_PLAN_REQUEST,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="learnity-gateway",
        value_deserializer=lambda v: __import__("json").loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    await consumer.start()
    logger.info("Kafka consumer started (topics: 5 learnity.* topics)")
    try:
        async for msg in consumer:
            event = msg.value
            event_type = event.get("event_type")
            handler = _HANDLERS.get(event_type)
            if handler is None:
                logger.warning("Unknown event_type=%s on topic=%s", event_type, msg.topic)
                continue
            try:
                await handler(event)
            except Exception:
                logger.exception("Error handling %s: %s", event_type, event)
    finally:
        await consumer.stop()
