"""
Интеграционные тесты сервиса профиля.
Используют реальный PostgreSQL (localhost:5432).

Запуск: pytest services/profile/tests/test_integration.py -v
Требование: docker compose up -d postgres && alembic upgrade head
"""

import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from shared.db import Base, get_db
from services.profile.main import app

TEST_DATABASE_URL = "postgresql+asyncpg://learnity:learnity@localhost:5432/learnity"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Каждый тест получает сессию в транзакции, которая откатывается после."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        # Запускаем каждый тест в savepoint — откатываем после без очистки таблиц
        async with async_session() as session:
            await session.begin_nested()
            yield session
            await session.rollback()

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    """HTTP-клиент с подменённой db-зависимостью."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def create_student(client: AsyncClient, grade: int = 8) -> uuid.UUID:
    resp = await client.post("/students", json={"grade": grade})
    assert resp.status_code == 201
    return uuid.UUID(resp.json()["student_id"])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Создание ученика
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_student(client):
    resp = await client.post("/students", json={"grade": 8})
    assert resp.status_code == 201
    data = resp.json()
    assert data["grade"] == 8
    assert "student_id" in data


@pytest.mark.asyncio
async def test_create_student_with_explicit_id(client):
    student_id = uuid.uuid4()
    resp = await client.post("/students", json={"grade": 7, "student_id": str(student_id)})
    assert resp.status_code == 201
    assert resp.json()["student_id"] == str(student_id)


@pytest.mark.asyncio
async def test_get_student_not_found(client):
    resp = await client.get(f"/students/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_student_returns_profile(client):
    student_id = await create_student(client, grade=9)
    resp = await client.get(f"/students/{student_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["grade"] == 9
    assert data["mastery"] == {}


# ---------------------------------------------------------------------------
# Mastery — начальное состояние
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mastery_empty_initially(client):
    student_id = await create_student(client)
    resp = await client.get(f"/students/{student_id}/mastery")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_mastery_single_kc_not_found(client):
    student_id = await create_student(client)
    resp = await client.get(f"/students/{student_id}/mastery/kc_pythagorean")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Отправка ответа — обновление mastery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_correct_answer_creates_mastery(client):
    student_id = await create_student(client)

    resp = await client.post(
        f"/students/{student_id}/interactions",
        json={
            "task_id": str(uuid.uuid4()),
            "part_id": "p1",
            "score": 1.0,
            "hints_used": 0,
            "primary_kcs": ["kc_pythagorean"],
        },
    )
    assert resp.status_code == 201
    updated = resp.json()["updated_mastery"]
    assert "kc_pythagorean" in updated
    assert updated["kc_pythagorean"] > 0.0


@pytest.mark.asyncio
async def test_correct_answer_increases_mastery(client):
    student_id = await create_student(client)
    kc = "kc_sqrt"

    # Первый правильный ответ
    resp1 = await client.post(
        f"/students/{student_id}/interactions",
        json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": [kc]},
    )
    prob1 = resp1.json()["updated_mastery"][kc]

    # Второй правильный ответ
    resp2 = await client.post(
        f"/students/{student_id}/interactions",
        json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": [kc]},
    )
    prob2 = resp2.json()["updated_mastery"][kc]

    assert prob2 > prob1


@pytest.mark.asyncio
async def test_incorrect_answers_give_less_mastery_than_correct(client):
    """
    Два ученика с одинаковой базой: один отвечает правильно, другой — неправильно.
    После N попыток у "правильного" mastery должен быть выше.

    Нюанс BKT: при mastery=0 даже неправильный ответ чуть повышает вероятность
    из-за transit P(T). Поэтому сравниваем накопленный эффект, а не первый шаг.
    """
    kc = "kc_algebra"
    student_correct = await create_student(client)
    student_incorrect = await create_student(client)

    for _ in range(5):
        await client.post(
            f"/students/{student_correct}/interactions",
            json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": [kc]},
        )
        await client.post(
            f"/students/{student_incorrect}/interactions",
            json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 0.0, "hints_used": 0, "primary_kcs": [kc]},
        )

    resp_c = await client.get(f"/students/{student_correct}/mastery/{kc}")
    resp_i = await client.get(f"/students/{student_incorrect}/mastery/{kc}")

    assert resp_c.json()["probability"] > resp_i.json()["probability"]


@pytest.mark.asyncio
async def test_hints_weaken_update(client):
    student_id_a = await create_student(client)
    student_id_b = await create_student(client)
    kc = "kc_fractions"

    resp_no_hints = await client.post(
        f"/students/{student_id_a}/interactions",
        json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": [kc]},
    )
    resp_with_hints = await client.post(
        f"/students/{student_id_b}/interactions",
        json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 3, "primary_kcs": [kc]},
    )

    prob_no_hints = resp_no_hints.json()["updated_mastery"][kc]
    prob_with_hints = resp_with_hints.json()["updated_mastery"][kc]
    assert prob_with_hints < prob_no_hints


@pytest.mark.asyncio
async def test_secondary_kcs_updated_weaker(client):
    student_id = await create_student(client)
    primary_kc = "kc_primary"
    secondary_kc = "kc_secondary"

    resp = await client.post(
        f"/students/{student_id}/interactions",
        json={
            "task_id": str(uuid.uuid4()),
            "part_id": "p1",
            "score": 1.0,
            "hints_used": 0,
            "primary_kcs": [primary_kc],
            "secondary_kcs": [secondary_kc],
        },
    )
    updated = resp.json()["updated_mastery"]
    assert updated[primary_kc] > updated[secondary_kc]


@pytest.mark.asyncio
async def test_mastery_persisted_after_interaction(client):
    student_id = await create_student(client)
    kc = "kc_geometry"

    await client.post(
        f"/students/{student_id}/interactions",
        json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": [kc]},
    )

    resp = await client.get(f"/students/{student_id}/mastery/{kc}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["kc_id"] == kc
    assert data["probability"] > 0.0
    assert data["attempts_count"] == 1


@pytest.mark.asyncio
async def test_attempts_count_increments(client):
    student_id = await create_student(client)
    kc = "kc_counting"

    for _ in range(3):
        await client.post(
            f"/students/{student_id}/interactions",
            json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": [kc]},
        )

    resp = await client.get(f"/students/{student_id}/mastery/{kc}")
    assert resp.json()["attempts_count"] == 3


@pytest.mark.asyncio
async def test_recent_accuracy_uses_raw_score_not_reward(client, db_session):
    student_id = await create_student(client)
    kc = "kc_accuracy_signal"

    await client.post(
        f"/students/{student_id}/interactions",
        json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": [kc]},
    )

    await db_session.execute(
        text(
            """
            INSERT INTO bandit_log
                (id, student_id, task_id, kc_id, context_vector, reward, raw_score, recommended_at)
            VALUES
                (:id, :student_id, :task_id, :kc_id, :context_vector, :reward, :raw_score, :recommended_at)
            """
        ),
        {
            "id": uuid.uuid4(),
            "student_id": student_id,
            "task_id": uuid.uuid4(),
            "kc_id": kc,
            "context_vector": [0.0] * 13,
            "reward": -0.25,
            "raw_score": 1.0,
            "recommended_at": datetime.utcnow(),
        },
    )
    await db_session.commit()

    await client.post(
        f"/students/{student_id}/interactions",
        json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": [kc]},
    )

    resp = await client.get(f"/students/{student_id}/mastery/{kc}")
    assert resp.status_code == 200
    assert resp.json()["confidence"] > 0.0


@pytest.mark.asyncio
async def test_probability_effective_leq_stored(client):
    """probability_effective <= probability (decay не может увеличить знание)."""
    student_id = await create_student(client)
    kc = "kc_trig"

    await client.post(
        f"/students/{student_id}/interactions",
        json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": [kc]},
    )

    resp = await client.get(f"/students/{student_id}/mastery/{kc}")
    data = resp.json()
    assert data["probability_effective"] <= data["probability"]


@pytest.mark.asyncio
async def test_submit_to_nonexistent_student(client):
    resp = await client.post(
        f"/students/{uuid.uuid4()}/interactions",
        json={"task_id": str(uuid.uuid4()), "part_id": "p1", "score": 1.0, "hints_used": 0, "primary_kcs": ["kc_x"]},
    )
    assert resp.status_code == 404
