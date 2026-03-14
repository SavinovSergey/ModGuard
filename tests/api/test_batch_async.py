"""Тесты асинхронного batch API (очередь + task_id, результат через GET /tasks)."""
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.task_store import TaskStore
from app.api.routes import router, get_task_store


@asynccontextmanager
async def _noop_lifespan(_app: FastAPI):
    """Пустой lifespan: без загрузки моделей (для тестов)."""
    yield


def _make_test_app() -> FastAPI:
    """Приложение только с API-роутером, без полного lifespan из main."""
    app = FastAPI(lifespan=_noop_lifespan)
    app.include_router(router, prefix=settings.api_prefix)
    return app


@pytest.fixture(scope="module")
def client_batch_async():
    """Клиент с in-memory task_store и замоканными очередью/БД (POST возвращает task_id)."""
    app = _make_test_app()
    store = TaskStore(None)
    app.dependency_overrides[get_task_store] = lambda: store

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def test_batch_async_returns_task_id(client_batch_async: TestClient):
    """POST /classify/batch-async возвращает task_id (при настроенных очереди и БД)."""
    with patch("app.api.routes.settings") as m:
        m.rabbitmq_url = "amqp://guest:guest@localhost:5672/"
        m.database_url = "postgresql://local/test"
        m.max_batch_size = 1000
        with patch("app.api.routes.create_task_pg"):
            with patch("app.api.routes.publish_task_request"):
                response = client_batch_async.post(
                    "/api/v1/classify/batch-async",
                    json={"items": [{"text": "нормальный комментарий"}]},
                )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert len(data["task_id"]) > 0


def test_get_task_not_found(client_batch_async: TestClient):
    """GET /tasks/{task_id} для несуществующего id возвращает 404."""
    with patch("app.api.routes.get_task_pg", return_value=None):
        response = client_batch_async.get("/api/v1/tasks/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_batch_async_then_get_task(client_batch_async: TestClient):
    """POST batch-async возвращает task_id; GET /tasks с моком get_task_pg возвращает completed с результатами."""
    mock_result = {
        "status": "completed",
        "results": [
            {"is_toxic": False, "toxicity_score": 0.0, "toxicity_types": {}, "tox_model_used": "regex", "spam_model_used": None, "is_spam": False, "spam_score": 0.0},
            {"is_toxic": True, "toxicity_score": 0.9, "toxicity_types": {}, "tox_model_used": "regex", "spam_model_used": None, "is_spam": False, "spam_score": 0.0},
        ],
        "error": None,
    }
    with patch("app.api.routes.settings") as m:
        m.rabbitmq_url = "amqp://x"
        m.database_url = "postgres://x"
        m.max_batch_size = 1000
        with patch("app.api.routes.create_task_pg"):
            with patch("app.api.routes.publish_task_request"):
                response = client_batch_async.post(
                    "/api/v1/classify/batch-async",
                    json={
                        "items": [
                            {"id": "a", "text": "нормальный текст"},
                            {"id": "b", "text": "ебать токсичный"},
                        ]
                    },
                )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    with patch("app.api.routes.get_task_pg", return_value=mock_result):
        get_resp = client_batch_async.get(f"/api/v1/tasks/{task_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["status"] == "completed"
    assert "results" in data
    assert len(data["results"]) == 2
    assert all("is_toxic" in r and "toxicity_score" in r for r in data["results"])


def test_batch_async_respects_max_batch_size(client_batch_async: TestClient):
    """batch-async возвращает 400 при превышении лимита размера батча."""
    with patch("app.api.routes.settings") as m:
        m.rabbitmq_url = "amqp://x"
        m.database_url = "postgres://x"
        m.max_batch_size = 1000
        items = [{"text": f"текст {i}"} for i in range(1001)]
        response = client_batch_async.post(
            "/api/v1/classify/batch-async",
            json={"items": items},
        )
    assert response.status_code == 400
    assert "exceeds maximum" in response.json()["detail"].lower()


def test_batch_async_empty_items_validation(client_batch_async: TestClient):
    """batch-async с пустым items возвращает 422."""
    response = client_batch_async.post(
        "/api/v1/classify/batch-async",
        json={"items": []},
    )
    assert response.status_code == 422


def test_classify_returns_task_id_with_queue(client_batch_async: TestClient):
    """POST /classify при настроенных очереди/БД возвращает task_id."""
    with patch("app.api.routes.settings") as m:
        m.rabbitmq_url = "amqp://x"
        m.database_url = "postgres://x"
        m.max_batch_size = 1000
        with patch("app.api.routes.create_task_pg"):
            with patch("app.api.routes.publish_task_request"):
                response = client_batch_async.post(
                    "/api/v1/classify",
                    json={"text": "нормальный комментарий"},
                )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert len(data["task_id"]) > 0
