"""Общие фикстуры тестов."""
from unittest.mock import AsyncMock, patch

import pytest


def _needs_api_lifespan_patch(request) -> bool:
    nodeid = request.node.nodeid.replace("\\", "/")
    if "tests/api/" not in nodeid:
        return False
    if "test_phase2_db" in nodeid:
        return False
    return True


@pytest.fixture(autouse=True)
def _patch_api_lifespan_deps(request):
    """
    В app.main lifespan: init_db, init_pool, init_queue_publisher.
    Только для tests/api/* без живого Postgres/RabbitMQ;
    реальная БД — test_phase2_db; unit-тесты app.main не трогаем.
    """
    if not _needs_api_lifespan_patch(request):
        yield
        return
    with patch("app.main.init_db", return_value=True):
        with patch("app.main.init_pool", new=AsyncMock(return_value=True)):
            with patch("app.main.init_queue_publisher", new=AsyncMock(return_value=True)):
                with patch("app.main.close_queue_publisher", new=AsyncMock()):
                    yield
