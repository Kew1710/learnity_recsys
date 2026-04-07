"""Kafka producer — публикация событий Learnity → backend."""

import json
import logging
import os
from datetime import datetime, timezone

from aiokafka import AIOKafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

TOPIC_TASK_RECOMMENDED = "learnity.task.recommended"
TOPIC_MASTERY_RESPONSE  = "learnity.mastery.response"
TOPIC_PLAN_CREATED      = "learnity.plan.created"

logger = logging.getLogger(__name__)

_producer: AIOKafkaProducer | None = None


async def start() -> None:
    global _producer
    _producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await _producer.start()
    logger.info("Kafka producer started (bootstrap=%s)", KAFKA_BOOTSTRAP)


async def stop() -> None:
    if _producer:
        await _producer.stop()


async def send(topic: str, payload: dict) -> None:
    if _producer is None:
        logger.warning("Kafka producer not started, skipping send to %s", topic)
        return
    payload.setdefault("schema_version", "v1")
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    await _producer.send_and_wait(topic, payload)
    logger.debug("Published to %s: %s", topic, payload)
