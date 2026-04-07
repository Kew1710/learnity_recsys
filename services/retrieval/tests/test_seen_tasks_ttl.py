"""Юнит-тесты seen_tasks TTL в retrieval/clients.py."""
import uuid
import pytest
import respx
import httpx

from services.retrieval.clients import get_seen_tasks

PROFILE_BASE = "http://127.0.0.1:8001"
STUDENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


@pytest.mark.asyncio
async def test_get_seen_tasks_no_ttl():
    """Без ttl_days — запрос без параметра ttl_days."""
    with respx.mock(base_url=PROFILE_BASE) as mock:
        mock.get(f"/students/{STUDENT_ID}/seen_tasks").mock(
            return_value=httpx.Response(200, json={"task_ids": ["task-1", "task-2"]})
        )
        async with httpx.AsyncClient() as client:
            result = await get_seen_tasks(client, STUDENT_ID)

        assert result == ["task-1", "task-2"]
        request = mock.calls[0].request
        assert "ttl_days" not in str(request.url)


@pytest.mark.asyncio
async def test_get_seen_tasks_with_ttl():
    """С ttl_days=90 — параметр передаётся в query string."""
    with respx.mock(base_url=PROFILE_BASE) as mock:
        mock.get(f"/students/{STUDENT_ID}/seen_tasks").mock(
            return_value=httpx.Response(200, json={"task_ids": ["task-1"]})
        )
        async with httpx.AsyncClient() as client:
            result = await get_seen_tasks(client, STUDENT_ID, ttl_days=90)

        assert result == ["task-1"]
        request = mock.calls[0].request
        assert "ttl_days=90" in str(request.url)


@pytest.mark.asyncio
async def test_get_seen_tasks_empty_list():
    """Пустой список при отсутствии истории."""
    with respx.mock(base_url=PROFILE_BASE) as mock:
        mock.get(f"/students/{STUDENT_ID}/seen_tasks").mock(
            return_value=httpx.Response(200, json={"task_ids": []})
        )
        async with httpx.AsyncClient() as client:
            result = await get_seen_tasks(client, STUDENT_ID, ttl_days=30)

    assert result == []


@pytest.mark.asyncio
async def test_get_seen_tasks_raises_on_error():
    """HTTP ошибка пробрасывается наружу."""
    with respx.mock(base_url=PROFILE_BASE) as mock:
        mock.get(f"/students/{STUDENT_ID}/seen_tasks").mock(
            return_value=httpx.Response(404)
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await get_seen_tasks(client, STUDENT_ID)


@pytest.mark.asyncio
async def test_get_seen_tasks_ttl_30():
    """Разные значения ttl_days корректно подставляются."""
    with respx.mock(base_url=PROFILE_BASE) as mock:
        mock.get(f"/students/{STUDENT_ID}/seen_tasks").mock(
            return_value=httpx.Response(200, json={"task_ids": ["task-x"]})
        )
        async with httpx.AsyncClient() as client:
            await get_seen_tasks(client, STUDENT_ID, ttl_days=30)

        request = mock.calls[0].request
        assert "ttl_days=30" in str(request.url)
