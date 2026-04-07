"""
Интеграционный тест Kafka-флоу: эмулирует backend и проверяет ответы Learnity.

Запуск (Learnity-сервисы должны быть подняты через `make dev`):
    python tools/test_kafka_integration.py

Что тестирует:
  1. STUDENT_REGISTERED → студент создаётся в Learnity (проверяем через HTTP)
  2. TASK_REQUESTED     → приходит TASK_RECOMMENDED с task_id и topic
  3. TASK_ANSWERED      → mastery обновляется (проверяем через HTTP)
  4. MASTERY_REQUESTED  → приходит MASTERY_RESPONSE с уровнями
"""

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone

import httpx
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

KAFKA_BOOTSTRAP = "localhost:9092"
GATEWAY_URL = "http://localhost:8005"
PROFILE_URL = "http://localhost:8001"

TOPIC_STUDENT_REGISTERED = "learnity.student.registered"
TOPIC_TASK_ANSWERED      = "learnity.task.answered"
TOPIC_TASK_REQUEST       = "learnity.task.request"
TOPIC_TASK_RECOMMENDED   = "learnity.task.recommended"
TOPIC_MASTERY_REQUEST    = "learnity.mastery.request"
TOPIC_MASTERY_RESPONSE   = "learnity.mastery.response"

STUDENT_ID = uuid.uuid4()
GRADE = 8


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()


async def publish(producer: AIOKafkaProducer, topic: str, payload: dict) -> None:
    payload.setdefault("schema_version", "v1")
    payload.setdefault("created_at", ts())
    await producer.send_and_wait(topic, json.dumps(payload).encode())
    print(f"  → [{topic}] {json.dumps(payload, ensure_ascii=False)[:120]}")


async def consume_one(topic: str, match_key: str, match_value: str, timeout: float = 30.0) -> dict | None:
    """Ждёт одно сообщение из topic где payload[match_key] == match_value.

    Использует earliest чтобы не пропустить сообщение из-за race condition;
    фильтрует по match_key чтобы игнорировать старые сообщения.
    """
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=f"test-{uuid.uuid4()}",
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
        return None
    finally:
        await consumer.stop()


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    sys.exit(1)


async def check_http(url: str, desc: str, retries: int = 5, delay: float = 1.0) -> dict:
    async with httpx.AsyncClient(trust_env=False) as http:
        for attempt in range(retries):
            try:
                resp = await http.get(url, timeout=5)
                if resp.status_code == 200:
                    ok(desc)
                    return resp.json()
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
            except httpx.RequestError:
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
        fail(f"{desc} → HTTP {resp.status_code}: {resp.text[:200]}")


# ─────────────────────────────────────────────────────────────────────────────


async def test_student_registered(producer: AIOKafkaProducer) -> None:
    print("\n[1] STUDENT_REGISTERED")
    await publish(producer, TOPIC_STUDENT_REGISTERED, {
        "event_type": "STUDENT_REGISTERED",
        "student_id": str(STUDENT_ID),
        "grade": GRADE,
    })

    # Даём Learnity время обработать
    await asyncio.sleep(3)

    # Проверяем что студент появился в Profile-сервисе
    await check_http(
        f"{PROFILE_URL}/students/{STUDENT_ID}",
        f"student {STUDENT_ID} создан в Profile",
    )


async def test_task_request(producer: AIOKafkaProducer) -> tuple[str, str]:
    """Возвращает (task_id, topic) из рекомендации."""
    print("\n[2] TASK_REQUESTED → TASK_RECOMMENDED")
    request_id = str(uuid.uuid4())

    # Слушаем TASK_RECOMMENDED ДО публикации запроса
    consume_task = asyncio.create_task(
        consume_one(TOPIC_TASK_RECOMMENDED, "request_id", request_id)
    )

    await publish(producer, TOPIC_TASK_REQUEST, {
        "event_type": "TASK_REQUESTED",
        "request_id": request_id,
        "student_id": str(STUDENT_ID),
        "subject": "algebra",
    })

    rec = await consume_task
    if not rec:
        fail("TASK_RECOMMENDED не пришёл за 20 сек")

    task_id = rec.get("task_id")
    topic = rec.get("topic")
    source = rec.get("recommendation_source")

    if not task_id or not topic:
        fail(f"TASK_RECOMMENDED не содержит task_id/topic: {rec}")

    ok(f"task_id={task_id}  topic={topic}  source={source}")
    return task_id, topic


async def test_task_answered(producer: AIOKafkaProducer, task_id: str, topic: str) -> None:
    print("\n[3] TASK_ANSWERED")
    await publish(producer, TOPIC_TASK_ANSWERED, {
        "event_type": "TASK_ANSWERED",
        "student_id": str(STUDENT_ID),
        "task_id": task_id,
        "score": 1.0,
        "hints_used": 0,
        "topic": topic,
        "subject": "algebra",
        "difficulty": "3",
        "recommendation_source": "zpd",
    })

    await asyncio.sleep(2)

    # Проверяем что mastery обновился
    data = await check_http(
        f"{PROFILE_URL}/students/{STUDENT_ID}/mastery/{topic}",
        f"mastery для {topic} обновлён",
    )
    mastery = data.get("probability_effective", data.get("probability", 0))
    ok(f"mastery({topic}) = {mastery:.3f}")


async def test_mastery_request(producer: AIOKafkaProducer, topic: str) -> None:
    print("\n[4] MASTERY_REQUESTED → MASTERY_RESPONSE")
    request_id = str(uuid.uuid4())

    consume_mastery = asyncio.create_task(
        consume_one(TOPIC_MASTERY_RESPONSE, "request_id", request_id)
    )

    await publish(producer, TOPIC_MASTERY_REQUEST, {
        "event_type": "MASTERY_REQUESTED",
        "request_id": request_id,
        "student_id": str(STUDENT_ID),
        "skill_keys": [topic],
    })

    resp = await consume_mastery
    if not resp:
        fail("MASTERY_RESPONSE не пришёл за 15 сек")

    levels = resp.get("levels", [])
    if not levels:
        fail(f"MASTERY_RESPONSE пустой: {resp}")

    for lvl in levels:
        ok(f"skill={lvl['skill_key']}  level={lvl['level']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────


ALL_TOPICS = [
    TOPIC_STUDENT_REGISTERED,
    TOPIC_TASK_ANSWERED,
    TOPIC_TASK_REQUEST,
    TOPIC_TASK_RECOMMENDED,
    TOPIC_MASTERY_REQUEST,
    TOPIC_MASTERY_RESPONSE,
    "learnity.plan.request",
    "learnity.plan.created",
    "learnity.alert.teacher",
]


async def ensure_topics() -> None:
    admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
    await admin.start()
    try:
        existing = set(await admin.list_topics())
        to_create = [
            NewTopic(t, num_partitions=3, replication_factor=1)
            for t in ALL_TOPICS if t not in existing
        ]
        if to_create:
            await admin.create_topics(to_create)
            print(f"  Created topics: {[t.name for t in to_create]}")
    finally:
        await admin.close()


async def main() -> None:
    print("=" * 60)
    print("Kafka Integration Test")
    print(f"student_id: {STUDENT_ID}  grade: {GRADE}")
    print("=" * 60)

    print("\n[0] Инициализация топиков")
    await ensure_topics()

    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
    await producer.start()
    try:
        await test_student_registered(producer)
        task_id, topic = await test_task_request(producer)
        await test_task_answered(producer, task_id, topic)
        await test_mastery_request(producer, topic)
    finally:
        await producer.stop()

    print("\n" + "=" * 60)
    print("✓ Все тесты прошли")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
