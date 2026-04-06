"""
Интеграционные тесты сервиса графа.
Используют реальный Neo4j (localhost:7688).

Запуск: pytest services/graph/tests/test_integration.py -v
Требование: docker compose up -d neo4j
"""

import os
os.environ["NEO4J_URI"] = "bolt://localhost:7688"

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from shared.neo4j_client import Neo4jClient
from services.graph.repository import GraphRepository
from services.graph.seed import KCS, EDGES
from services.graph.main import app, neo4j_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="module", autouse=True)
async def seed_graph():
    """Один раз на модуль: подключаемся, заливаем seed, после — чистим."""
    test_client = Neo4jClient()
    test_client._driver = None
    from neo4j import AsyncGraphDatabase
    test_client._driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7688", auth=("neo4j", "learnity123")
    )
    await test_client._driver.verify_connectivity()

    repo = GraphRepository(test_client)
    await repo.clear()
    for kc in KCS:
        await repo.create_kc(kc)
    for from_kc, to_kc, strength in EDGES:
        await repo.create_prerequisite(from_kc, to_kc, strength)

    yield

    await repo.clear()
    await test_client._driver.close()


@pytest_asyncio.fixture(scope="module")
async def client():
    """HTTP-клиент с подключённым neo4j_client."""
    from neo4j import AsyncGraphDatabase
    neo4j_client._driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7688", auth=("neo4j", "learnity123")
    )
    await neo4j_client._driver.verify_connectivity()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    await neo4j_client._driver.close()
    neo4j_client._driver = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /nodes/{kc_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_existing_node(client):
    resp = await client.get("/nodes/kc_pythagorean_know")
    assert resp.status_code == 200
    data = resp.json()
    assert data["kc_id"] == "kc_pythagorean_know"
    assert data["grade_introduced"] == 8


@pytest.mark.asyncio
async def test_get_nonexistent_node(client):
    resp = await client.get("/nodes/kc_does_not_exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /nodes/{kc_id}/prerequisites
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prerequisites_of_pythagorean_know(client):
    resp = await client.get("/nodes/kc_pythagorean_know/prerequisites")
    assert resp.status_code == 200
    prereqs = resp.json()
    kc_ids = [p["kc_id"] for p in prereqs]
    # Из seed: три пререквизита у kc_pythagorean_know
    assert "kc_square_power" in kc_ids
    assert "kc_right_triangle_parts" in kc_ids
    assert "kc_sqrt_concept" in kc_ids


@pytest.mark.asyncio
async def test_prerequisites_of_root_node_empty(client):
    """kc_natural_numbers не имеет пререквизитов."""
    resp = await client.get("/nodes/kc_natural_numbers/prerequisites")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_prerequisites_of_nonexistent_node(client):
    resp = await client.get("/nodes/kc_unknown/prerequisites")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /zpd
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_zpd_new_student_gets_candidates(client):
    """Новый ученик (mastery=0 везде) должен получить базовые KC."""
    resp = await client.post("/zpd", json={"mastery": {}, "student_grade": 8})
    assert resp.status_code == 200
    candidates = resp.json()
    assert len(candidates) > 0


@pytest.mark.asyncio
async def test_zpd_mastered_kc_excluded(client):
    """Освоенная KC не должна появляться в ZPD."""
    resp = await client.post("/zpd", json={
        "mastery": {"kc_natural_numbers": 0.95},
        "student_grade": 8,
    })
    kc_ids = [c["kc_id"] for c in resp.json()]
    assert "kc_natural_numbers" not in kc_ids


@pytest.mark.asyncio
async def test_zpd_ready_flag_correct(client):
    """kc_fractions_operations (grade=6): пререквизиты grade=5 < floor=6 → assumed mastered → ready=True."""
    resp = await client.post("/zpd", json={"mastery": {}, "student_grade": 8})
    node = next((c for c in resp.json() if c["kc_id"] == "kc_fractions_operations"), None)
    assert node is not None
    assert node["ready"] is True


@pytest.mark.asyncio
async def test_zpd_unmet_prereqs_not_ready(client):
    """kc_pythagorean_know с неосвоенными пререквизитами — ready=False."""
    resp = await client.post("/zpd", json={"mastery": {}, "student_grade": 8})
    pyth = next((c for c in resp.json() if c["kc_id"] == "kc_pythagorean_know"), None)
    assert pyth is not None
    assert pyth["ready"] is False


@pytest.mark.asyncio
async def test_zpd_prereqs_met_makes_ready(client):
    """Если все пререквизиты kc_pythagorean_know освоены — ready=True."""
    mastery = {
        "kc_square_power":         0.9,
        "kc_sqrt_concept":         0.9,
        "kc_right_triangle_parts": 0.9,
    }
    resp = await client.post("/zpd", json={"mastery": mastery, "student_grade": 8})
    pyth = next((c for c in resp.json() if c["kc_id"] == "kc_pythagorean_know"), None)
    assert pyth is not None
    assert pyth["ready"] is True


@pytest.mark.asyncio
async def test_zpd_grade_filter(client):
    """Ученик 8 класса не должен видеть KC 4 класса (grade < 8-2=6)."""
    resp = await client.post("/zpd", json={"mastery": {}, "student_grade": 8})
    kc_ids = [c["kc_id"] for c in resp.json()]
    assert "kc_natural_numbers" not in kc_ids  # grade=4, ниже порога


@pytest.mark.asyncio
async def test_zpd_ready_kcs_come_first(client):
    """ready=True KC идут перед ready=False в ответе."""
    resp = await client.post("/zpd", json={"mastery": {}, "student_grade": 8})
    candidates = resp.json()
    ready_indices = [i for i, c in enumerate(candidates) if c["ready"]]
    not_ready_indices = [i for i, c in enumerate(candidates) if not c["ready"]]
    if ready_indices and not_ready_indices:
        assert max(ready_indices) < min(not_ready_indices)


# ---------------------------------------------------------------------------
# GET /path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_path_exists(client):
    resp = await client.get("/path?from_kc=kc_natural_numbers&to_kc=kc_pythagorean_know")
    assert resp.status_code == 200
    path = resp.json()["path"]
    assert path[0] == "kc_natural_numbers"
    assert path[-1] == "kc_pythagorean_know"
    assert len(path) > 2


@pytest.mark.asyncio
async def test_path_not_found(client):
    """Пути от pythagorean_know до natural_numbers нет (граф однонаправленный)."""
    resp = await client.get("/path?from_kc=kc_pythagorean_know&to_kc=kc_natural_numbers")
    assert resp.status_code == 404
