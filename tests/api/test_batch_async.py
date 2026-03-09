"""Тесты асинхронного batch API и кэша.

Используется отдельное тестовое приложение с пустым lifespan, чтобы не загружать
PyTorch/модели (избегаем segfault в torch._dynamo при повторных циклах).
"""
import time
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.model_manager import ModelManager
from app.core.cache import NoOpModerationCache
from app.core.task_store import TaskStore
from app.services.classification import ClassificationService
from app.models.toxicity.regex_model import RegexModel
from app.api.routes import (
    router,
    get_classification_service,
    get_model_manager,
    get_task_store,
)


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
    """Клиент с regex, in-memory task_store, без Redis и без загрузки PyTorch."""
    app = _make_test_app()

    test_model_manager = ModelManager()
    regex_model = RegexModel()
    regex_model.load()
    test_model_manager.register_model("regex", regex_model)
    test_model_manager.set_current_model("regex")
    cache = NoOpModerationCache()
    test_classification_service = ClassificationService(
        test_model_manager,
        moderation_cache=cache,
    )
    store = TaskStore(None)

    app.dependency_overrides[get_classification_service] = lambda: test_classification_service
    app.dependency_overrides[get_model_manager] = lambda: test_model_manager
    app.dependency_overrides[get_task_store] = lambda: store

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def test_batch_async_returns_task_id(client_batch_async: TestClient):
    """POST /classify/batch-async возвращает task_id."""
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
    response = client_batch_async.get("/api/v1/tasks/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_batch_async_then_get_task(client_batch_async: TestClient):
    """После POST batch-async GET по task_id возвращает статус и при ожидании — completed с результатами."""
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

    # Фоновая задача может выполниться не сразу; даём время
    for _ in range(15):
        get_resp = client_batch_async.get(f"/api/v1/tasks/{task_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["status"] in ("queued", "processing", "completed", "failed")
        if data["status"] == "completed":
            assert "results" in data
            assert len(data["results"]) == 2
            assert all("is_toxic" in r and "toxicity_score" in r for r in data["results"])
            break
        time.sleep(0.2)
    else:
        pytest.fail("Task did not complete within timeout")


def test_batch_async_respects_max_batch_size(client_batch_async: TestClient):
    """batch-async возвращает 400 при превышении лимита размера батча."""
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


def test_classify_with_cache_stub(client_batch_async: TestClient):
    """Классификация работает с no-op кэшем (без Redis)."""
    response = client_batch_async.post(
        "/api/v1/classify",
        json={"text": "нормальный комментарий"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data
    assert "toxicity_score" in data
    assert data.get("model_used") == "regex"
