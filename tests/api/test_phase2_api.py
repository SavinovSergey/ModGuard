"""Тесты Фазы 2: API читает результат из Postgres (мок get_task_pg) и fallback на task_store."""
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.task_store import TaskStore
from app.api.routes import get_task_store


@pytest.fixture
def client_phase2():
    """Клиент с переопределённым task_store для GET /tasks (очередь/бэкенд не инжектятся в API)."""
    store = TaskStore(None)
    app.dependency_overrides[get_task_store] = lambda: store

    with TestClient(app) as client:
        yield client, store
    app.dependency_overrides.clear()


def test_get_task_reads_from_postgres_when_configured(client_phase2):
    """GET /tasks/{task_id} возвращает данные из get_task_pg, когда он что-то возвращает."""
    client, _ = client_phase2
    task_id = str(uuid.uuid4())
    mock_result = {
        "status": "completed",
        "results": [
            {"is_toxic": False, "toxicity_score": 0.0, "toxicity_types": {}, "tox_model_used": "regex", "spam_model_used": None},
            {"is_toxic": True, "toxicity_score": 0.9, "toxicity_types": {"x": 0.9}, "tox_model_used": "regex", "spam_model_used": None},
        ],
        "error": None,
    }
    # Маршрут вызывает get_task_pg только при settings.database_url; подменяем атрибут
    from app.api import routes
    with patch.object(routes.settings, "database_url", "postgresql://local"):
        with patch("app.api.routes.get_task_pg", return_value=mock_result):
            response = client.get(f"/api/v1/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["results"] is not None
    assert len(data["results"]) == 2
    assert data["results"][0]["is_toxic"] is False
    assert data["results"][1]["is_toxic"] is True
    assert data["results"][1]["toxicity_score"] == 0.9


def test_get_task_fallback_to_task_store_when_postgres_returns_none(client_phase2):
    """GET /tasks/{task_id}: при get_task_pg()=None читается task_store (Phase 1 путь)."""
    client, store = client_phase2
    task_id = str(uuid.uuid4())
    store.create_task(task_id, [{"text": "hello"}])
    store.set_task_result(
        task_id,
        [{"is_toxic": False, "toxicity_score": 0.0, "toxicity_types": {}, "tox_model_used": "regex", "spam_model_used": None}],
        status="completed",
    )
    with patch("app.api.routes.get_task_pg", return_value=None):
        response = client.get(f"/api/v1/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["results"] is not None
    assert len(data["results"]) == 1


def test_get_task_404_when_both_postgres_and_store_empty(client_phase2):
    """GET /tasks/{task_id} возвращает 404, если задачи нет ни в Postgres, ни в store."""
    client, _ = client_phase2
    with patch("app.api.routes.get_task_pg", return_value=None):
        response = client.get("/api/v1/tasks/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
