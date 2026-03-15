"""Тесты Фазы 3: поля is_spam/spam_score в ответе GET /tasks (и в ClassificationService)."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.model_manager import ModelManager
from app.services.classification import ClassificationService
from app.models.toxicity.regex_model import RegexModel
from app.api.routes import get_task_store
from app.core.task_store import TaskStore


@pytest.fixture
def client_phase3():
    """Клиент с task_store (API только очередь + GET /tasks)."""
    store = TaskStore(None)
    app.dependency_overrides[get_task_store] = lambda: store

    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_classify_response_has_spam_fields(client_phase3: TestClient):
    """GET /tasks/{task_id} с completed результатом содержит is_spam и spam_score."""
    mock_result = {
        "status": "completed",
        "results": [
            {
                "is_toxic": False,
                "toxicity_score": 0.0,
                "toxicity_types": {},
                "tox_model_used": "regex",
                "spam_model_used": None,
                "is_spam": False,
                "spam_score": 0.0,
            }
        ],
        "error": None,
    }
    with patch("app.api.routes.settings") as m:
        m.rabbitmq_url = "amqp://x"
        m.database_url = "postgres://x"
        m.max_batch_size = 1000
        with patch("app.api.routes.create_task_pg"):
            with patch("app.api.routes.publish_task_request"):
                resp = client_phase3.post(
                    "/api/v1/classify",
                    json={"text": "нормальный комментарий"},
                )
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    with patch("app.api.routes.get_task_pg", return_value=mock_result):
        response = client_phase3.get(f"/api/v1/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["results"] is not None
    assert len(data["results"]) == 1
    r = data["results"][0]
    assert "is_toxic" in r
    assert "is_spam" in r
    assert "spam_score" in r
    assert r["is_spam"] is False
    assert r["spam_score"] == 0.0


def test_classify_batch_response_has_spam_fields(client_phase3: TestClient):
    """GET /tasks/{task_id} для батча содержит is_spam и spam_score в каждом элементе."""
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
                resp = client_phase3.post(
                    "/api/v1/classify/batch-async",
                    json={"items": [{"text": "текст один"}, {"text": "ебать второй"}]},
                )
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    with patch("app.api.routes.get_task_pg", return_value=mock_result):
        response = client_phase3.get(f"/api/v1/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert "is_spam" in r
        assert "spam_score" in r
    assert data["results"][0]["is_toxic"] is False
    assert data["results"][1]["is_toxic"] is True


def test_phase3_parallel_merge():
    """Классификация объединяет результат токсичности и спама (спам — заглушка)."""
    model_manager = ModelManager()
    regex_model = RegexModel()
    regex_model.load()
    model_manager.register_model("regex", regex_model)
    model_manager.set_current_model("regex")
    service = ClassificationService(model_manager, spam_model=None)
    result = service.classify("ебать тест")
    assert result["is_toxic"] is True
    assert result["is_spam"] is False
    assert result["spam_score"] == 0.0
    batch = service.classify_batch(["норм", "тоже норм"])
    assert len(batch) == 2
    for r in batch:
        assert "is_spam" in r and "spam_score" in r
