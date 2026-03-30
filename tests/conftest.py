"""Общие фикстуры тестов."""
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _patch_api_init_db_outside_phase2(request):
    """
    В app.main lifespan вызывается init_db при заданном DATABASE_URL из .env.
    Для большинства тестов нужен запуск API без живого Postgres; реальная БД — только test_phase2_db.
    """
    if "test_phase2_db" in request.node.nodeid:
        yield
        return
    with patch("app.main.init_db", return_value=True):
        yield
