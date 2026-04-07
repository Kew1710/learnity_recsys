import json
import os
from typing import Callable, Awaitable

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from pydantic import BaseModel

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

TOPIC_ANSWER_SUBMITTED = "answer_submitted"
TOPIC_MASTERY_UPDATED = "mastery_updated"


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

class KafkaProducerClient:
    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def publish(self, topic: str, event: BaseModel) -> None:
        assert self._producer, "Producer not started"
        await self._producer.send_and_wait(topic, event.model_dump(mode="json"))

    # context manager support
    async def __aenter__(self) -> "KafkaProducerClient":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------

class KafkaConsumerClient:
    def __init__(self, topics: list[str], group_id: str) -> None:
        self._topics = topics
        self._group_id = group_id
        self._consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            *self._topics,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=self._group_id,
            value_deserializer=lambda v: json.loads(v.decode()),
            auto_offset_reset="earliest",
        )
        await self._consumer.start()

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()

    async def consume(self, handler: Callable[[str, dict], Awaitable[None]]) -> None:
        """Бесконечный цикл — вызывает handler(topic, message_dict) для каждого сообщения."""
        assert self._consumer, "Consumer not started"
        async for msg in self._consumer:
            await handler(msg.topic, msg.value)

    async def __aenter__(self) -> "KafkaConsumerClient":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()
