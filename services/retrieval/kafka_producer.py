"""Kafka producer для публикации MicroSummary событий."""

import json
import logging
import os

from aiokafka import AIOKafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_MICRO_SUMMARIES = "micro_summaries"

logger = logging.getLogger(__name__)

_producer: AIOKafkaProducer | None = None


async def get_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is None:
        _producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await _producer.start()
    return _producer


async def publish_micro_summary(summary: dict) -> None:
    """Публикует MicroSummary в топик micro_summaries.
    При недоступности Kafka — логирует и продолжает (не блокирует рекомендацию).
    """
    try:
        producer = await get_producer()
        await producer.send_and_wait(TOPIC_MICRO_SUMMARIES, summary)
    except Exception as e:
        logger.warning("Kafka unavailable, micro_summary not published: %s", e)
