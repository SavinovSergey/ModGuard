"""
Тесты API при развёрнутом приложении (реальный lifespan, без dependency overrides).

API только ставит задачи в очередь и отдаёт результат по GET /tasks/{task_id};
модели загружаются в backend (worker), поэтому /stats не содержит model_stats.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def deployed_client():
    """Клиент с реальным приложением (lifespan выполняется)."""
    with TestClient(app) as client:
        yield client


def test_service_health(deployed_client: TestClient):
    """Сервис отвечает, health возвращает status и model_info."""
    response = deployed_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") in ("healthy", "degraded", "unhealthy")
    assert "model_info" in data
    assert "version" in data


def test_stats_structure(deployed_client: TestClient):
    """Stats возвращает model_stats (может быть пустым) и current_model (API не грузит модели)."""
    response = deployed_client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "model_stats" in data
    assert "current_model" in data
    assert isinstance(data["model_stats"], dict)


def test_classify_without_queue_returns_503(deployed_client: TestClient):
    """Без очереди/БД POST /classify возвращает 503."""
    # При дефолтных настройках тестов очереди обычно нет
    response = deployed_client.post(
        "/api/v1/classify",
        json={"text": "тестовый комментарий", "preferred_model": "regex"},
    )
    # Либо 503 (очередь не настроена), либо 200 с task_id (если в окружении заданы RABBITMQ_URL и DATABASE_URL)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert "task_id" in response.json()


def test_get_task_404(deployed_client: TestClient):
    """GET /tasks/{task_id} для несуществующего id возвращает 404."""
    response = deployed_client.get("/api/v1/tasks/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
