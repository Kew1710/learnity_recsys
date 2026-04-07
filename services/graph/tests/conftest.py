import asyncio
import pytest


@pytest.fixture(scope="module")
def event_loop():
    """Один event loop на весь модуль — нужен для module-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
