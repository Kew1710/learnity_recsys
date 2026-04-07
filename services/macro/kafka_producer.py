"""Kafka producer для macro-сервиса — публикация teacher_alert событий."""

import json
import logging
import os
from datetime import datetime, timezone

from aiokafka import AIOKafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_TEACHER_ALERT = "learnity.alert.teacher"

logger = logging.getLogger(__name__)

_producer: AIOKafkaProducer | None = None


async def start() -> None:
    global _producer
    _producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await _producer.start()
    logger.info("Macro Kafka producer started")


async def stop() -> None:
    if _producer:
        await _producer.stop()


async def send_teacher_alert(
    student_id: str,
    alert_type: str,
    skill_key: str,
    mastery_at_alert: float,
    tasks_spent: int,
    message: str,
) -> None:
    if _producer is None:
        logger.warning("Macro Kafka producer not started, skipping teacher alert")
        return
    payload = {
        "event_type": "TEACHER_ALERT",
        "student_id": student_id,
        "alert_type": alert_type,
        "skill_key": skill_key,
        "mastery_at_alert": mastery_at_alert,
        "tasks_spent": tasks_spent,
        "message": message,
        "schema_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await _producer.send_and_wait(TOPIC_TEACHER_ALERT, payload)
    logger.debug("Teacher alert published: student=%s type=%s kc=%s", student_id, alert_type, skill_key)
