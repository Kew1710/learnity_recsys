"""
Интеграционные тесты сервиса банка заданий.
Используют реальный PostgreSQL (localhost:5432) со stub-данными из seed.

Запуск: pytest services/task_bank/tests/test_integration.py -v
Требование: docker compose up -d postgres && alembic upgrade head && python -m services.task_bank.seed
"""

import uuid
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from shared.db import Base, get_db
from services.task_bank.main import app

TEST_DATABASE_URL = "postgresql+asyncpg://learnity:learnity@localhost:5432/learnity"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="function")
async def db_session():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        async with async_session() as session:
            await session.begin_nested()
            yield session
            await session.rollback()
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /tasks/by_kc
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_by_kc_returns_stubs(client):
    resp = await client.get("/tasks/by_kc", params={"kc_id": "kc_natural_numbers", "grade_min": 5})
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) == 5
    for t in tasks:
        primary_kcs_all = [kc for p in t["parts"] for kc in p["primary_kcs"]]
        assert "kc_natural_numbers" in primary_kcs_all


@pytest.mark.asyncio
async def test_by_kc_grade_filter(client):
    # kc_pythagorean_know введена в 8 классе — не появляется для grade_min=7
    resp = await client.get("/tasks/by_kc", params={"kc_id": "kc_pythagorean_know", "grade_min": 7})
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_by_kc_grade_filter_passes(client):
    resp = await client.get("/tasks/by_kc", params={"kc_id": "kc_pythagorean_know", "grade_min": 8})
    assert resp.status_code == 200
    assert len(resp.json()) == 5


@pytest.mark.asyncio
async def test_by_kc_exclude_filter(client):
    # Получаем все задания по KC
    resp_all = await client.get("/tasks/by_kc", params={"kc_id": "kc_fractions_basic", "grade_min": 5})
    all_tasks = resp_all.json()
    assert len(all_tasks) == 5

    # Исключаем первые 3
    exclude_ids = [t["task_id"] for t in all_tasks[:3]]
    resp_filtered = await client.get(
        "/tasks/by_kc",
        params={"kc_id": "kc_fractions_basic", "grade_min": 5, "exclude": exclude_ids},
    )
    filtered = resp_filtered.json()
    assert len(filtered) == 2
    filtered_ids = {t["task_id"] for t in filtered}
    assert filtered_ids.isdisjoint(set(exclude_ids))


@pytest.mark.asyncio
async def test_by_kc_unknown_kc_returns_empty(client):
    resp = await client.get("/tasks/by_kc", params={"kc_id": "kc_nonexistent", "grade_min": 8})
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_task_by_id(client):
    # Берём любое задание через by_kc
    resp = await client.get("/tasks/by_kc", params={"kc_id": "kc_quadratic_eq", "grade_min": 8})
    task_id = resp.json()[0]["task_id"]

    resp2 = await client.get(f"/tasks/{task_id}")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["task_id"] == task_id
    assert len(data["parts"]) == 1
    assert data["parts"][0]["primary_kcs"] == ["kc_quadratic_eq"]


@pytest.mark.asyncio
async def test_get_task_not_found(client):
    resp = await client.get(f"/tasks/{uuid.uuid4()}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Структура stub-задания
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stub_structure(client):
    resp = await client.get("/tasks/by_kc", params={"kc_id": "kc_derivative_concept", "grade_min": 10})
    tasks = resp.json()
    assert len(tasks) == 5

    difficulties = sorted(t["parts"][0]["irt_difficulty"] for t in tasks)
    # Сложности должны возрастать (5 разных уровней)
    assert difficulties == sorted(set(difficulties))
    # Все в допустимом диапазоне
    assert all(0.05 <= d <= 0.95 for d in difficulties)


@pytest.mark.asyncio
async def test_stub_description_format(client):
    resp = await client.get("/tasks/by_kc", params={"kc_id": "kc_circle_length_area", "grade_min": 6})
    for task in resp.json():
        desc = task["parts"][0]["description"]
        assert desc.startswith("[STUB]")


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_count_by_kc(client):
    resp = await client.get("/tasks/by_kc/count", params={"kc_id": "kc_linear_eq_1var"})
    assert resp.status_code == 200
    assert resp.json()["count"] == 5
