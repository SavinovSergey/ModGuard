"""Интеграционные тесты для API endpoints (очередь + task_id, классификация в backend)."""
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import get_task_store
from app.core.task_store import TaskStore


@pytest.fixture
def client():
    """Тестовый клиент без переопределений (API только очередь + GET /tasks)."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_with_queue(client):
    """Клиент с замоканными очередью и БД: POST возвращает task_id, GET /tasks — по моку get_task_pg."""
    with patch("app.api.routes.settings") as mock_settings:
        mock_settings.rabbitmq_url = "amqp://guest:guest@localhost:5672/"
        mock_settings.database_url = "postgresql://local/test"
        mock_settings.max_batch_size = 1000
        with patch("app.api.routes.create_task_pg"):
            with patch("app.api.routes.publish_task_request"):
                yield client


def _completed_result(is_toxic=False, toxicity_score=0.0, tox_model_used="regex"):
    return {
        "status": "completed",
        "results": [
            {
                "is_toxic": is_toxic,
                "toxicity_score": toxicity_score,
                "toxicity_types": {},
                "tox_model_used": tox_model_used,
                "spam_model_used": None,
                "is_spam": False,
                "spam_score": 0.0,
            }
        ],
        "error": None,
    }


def test_root_endpoint(client):
    """Тест корневого endpoint"""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ToxicFilter"
    assert data["status"] == "running"
    assert "version" in data


def test_classify_returns_task_id(client_with_queue):
    """POST /classify ставит задачу в очередь и возвращает task_id."""
    response = client_with_queue.post(
        "/api/v1/classify",
        json={"text": "это нормальный комментарий"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert len(data["task_id"]) > 0


def test_classify_then_get_task_result(client_with_queue):
    """POST /classify возвращает task_id; GET /tasks/{task_id} с моком возвращает результат."""
    task_id = str(uuid.uuid4())
    mock_result = _completed_result(is_toxic=False, toxicity_score=0.0)

    with patch("app.api.routes.create_task_pg") as mock_create:
        mock_create.return_value = None
        response = client_with_queue.post(
            "/api/v1/classify",
            json={"text": "нормальный комментарий"},
        )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    with patch("app.api.routes.get_task_pg", return_value=mock_result):
        get_resp = client_with_queue.get(f"/api/v1/tasks/{task_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["status"] == "completed"
    assert data["results"] is not None
    assert len(data["results"]) == 1
    assert data["results"][0]["is_toxic"] is False
    assert "toxicity_score" in data["results"][0]
    assert data["results"][0].get("tox_model_used") == "regex"


def test_classify_toxic_result_via_get_task(client_with_queue):
    """Результат с is_toxic=True приходит через GET /tasks/{task_id}."""
    task_id = str(uuid.uuid4())
    mock_result = _completed_result(is_toxic=True, toxicity_score=0.9)

    with patch("app.api.routes.create_task_pg"):
        resp = client_with_queue.post(
            "/api/v1/classify",
            json={"text": "иди нахуй ебать"},
        )
    task_id = resp.json()["task_id"]

    with patch("app.api.routes.get_task_pg", return_value=mock_result):
        get_resp = client_with_queue.get(f"/api/v1/tasks/{task_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["results"][0]["is_toxic"] is True
    assert data["results"][0]["toxicity_score"] == 0.9


def test_classify_with_preferred_model_returns_task_id(client_with_queue):
    """POST /classify с preferred_model возвращает task_id (модель применяется в backend)."""
    response = client_with_queue.post(
        "/api/v1/classify",
        json={"text": "тестовый текст", "preferred_model": "regex"},
    )
    assert response.status_code == 200
    assert "task_id" in response.json()


def test_classify_with_context_returns_task_id(client_with_queue):
    """POST /classify с context возвращает task_id."""
    response = client_with_queue.post(
        "/api/v1/classify",
        json={
            "text": "это ответ на предыдущее сообщение",
            "context": ["предыдущее сообщение", "еще одно"],
        },
    )
    assert response.status_code == 200
    assert "task_id" in response.json()


def test_classify_empty_text_returns_task_id(client_with_queue):
    """POST /classify с пустым текстом возвращает task_id (результат — в GET)."""
    response = client_with_queue.post(
        "/api/v1/classify",
        json={"text": ""},
    )
    assert response.status_code == 200
    assert "task_id" in response.json()


def test_classify_batch_returns_task_id(client_with_queue):
    """POST /classify/batch ставит батч в очередь и возвращает task_id."""
    response = client_with_queue.post(
        "/api/v1/classify/batch",
        json={
            "texts": [
                "нормальный комментарий",
                "токсичный ебать комментарий",
                "спасибо за помощь",
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert len(data["task_id"]) > 0


def test_classify_batch_then_get_results(client_with_queue):
    """POST /classify/batch → task_id; GET /tasks возвращает батч результатов."""
    mock_result = {
        "status": "completed",
        "results": [
            {"is_toxic": False, "toxicity_score": 0.0, "toxicity_types": {}, "tox_model_used": "regex", "spam_model_used": None, "is_spam": False, "spam_score": 0.0},
            {"is_toxic": True, "toxicity_score": 0.9, "toxicity_types": {}, "tox_model_used": "regex", "spam_model_used": None, "is_spam": False, "spam_score": 0.0},
            {"is_toxic": False, "toxicity_score": 0.0, "toxicity_types": {}, "tox_model_used": "regex", "spam_model_used": None, "is_spam": False, "spam_score": 0.0},
        ],
        "error": None,
    }
    with patch("app.api.routes.create_task_pg"):
        resp = client_with_queue.post(
            "/api/v1/classify/batch",
            json={"texts": ["a", "b", "c"]},
        )
    task_id = resp.json()["task_id"]
    with patch("app.api.routes.get_task_pg", return_value=mock_result):
        get_resp = client_with_queue.get(f"/api/v1/tasks/{task_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert len(data["results"]) == 3
    assert data["results"][1]["is_toxic"] is True
    assert 0.0 <= data["results"][0]["toxicity_score"] <= 1.0


def test_classify_batch_with_preferred_model_returns_task_id(client_with_queue):
    """POST /classify/batch с preferred_model возвращает task_id."""
    response = client_with_queue.post(
        "/api/v1/classify/batch",
        json={"texts": ["текст 1", "текст 2"], "preferred_model": "regex"},
    )
    assert response.status_code == 200
    assert "task_id" in response.json()


def test_classify_batch_empty_list(client):
    """POST /classify/batch с пустым списком — 422."""
    with patch("app.api.routes.settings") as m:
        m.rabbitmq_url = "amqp://x"
        m.database_url = "postgres://x"
        m.max_batch_size = 1000
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": []},
        )
    assert response.status_code == 422


def test_classify_batch_too_large(client_with_queue):
    """POST /classify/batch с превышением лимита — 400."""
    large_texts = [f"текст {i}" for i in range(1001)]
    with patch("app.api.routes.settings") as m:
        m.rabbitmq_url = "amqp://x"
        m.database_url = "postgres://x"
        m.max_batch_size = 1000
        response = client_with_queue.post(
            "/api/v1/classify/batch",
            json={"texts": large_texts},
        )
    assert response.status_code == 400
    assert "exceeds maximum" in response.json()["detail"].lower()


def test_health_endpoint(client):
    """Тест health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_info" in data
    assert data["status"] in ["healthy", "degraded", "unhealthy"]


def test_health_endpoint_model_info_note(client):
    """API health возвращает model_info (классификация в backend)."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    model_info = response.json().get("model_info") or {}
    assert "note" in model_info
    assert "backend" in model_info["note"].lower() or "worker" in model_info["note"].lower()


def test_stats_endpoint(client):
    """API stats возвращает model_stats и current_model (могут быть пустыми)."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "model_stats" in data
    assert "current_model" in data
    assert isinstance(data["model_stats"], dict)


def test_classify_without_queue_returns_503(client):
    """Без очереди/БД POST /classify возвращает 503."""
    with patch("app.api.routes.settings") as m:
        m.rabbitmq_url = None
        m.database_url = None
        response = client.post(
            "/api/v1/classify",
            json={"text": "любой текст"},
        )
    assert response.status_code == 503


def test_classify_invalid_json(client_with_queue):
    """Невалидный JSON для /classify — 422."""
    response = client_with_queue.post(
        "/api/v1/classify",
        json={"invalid": "data"},
    )
    assert response.status_code == 422


def test_classify_missing_text_field(client):
    """Запрос без поля text — 422."""
    response = client.post(
        "/api/v1/classify",
        json={},
    )
    assert response.status_code == 422


def test_get_task_404(client):
    """GET /tasks/{task_id} для несуществующего id — 404."""
    with patch("app.api.routes.get_task_pg", return_value=None):
        response = client.get("/api/v1/tasks/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_multiple_classify_requests_return_task_ids(client_with_queue):
    """Несколько POST /classify возвращают разные task_id."""
    texts = ["нормальный комментарий", "токсичный ебать", "спасибо", "иди нахуй"]
    task_ids = []
    for text in texts:
        response = client_with_queue.post(
            "/api/v1/classify",
            json={"text": text},
        )
        assert response.status_code == 200
        task_ids.append(response.json()["task_id"])
    assert len(set(task_ids)) == len(task_ids)


def test_classify_special_characters_returns_task_id(client_with_queue):
    """POST /classify со спецсимволами возвращает task_id."""
    response = client_with_queue.post(
        "/api/v1/classify",
        json={"text": "Текст с эмодзи 😀 и спецсимволами !@#$%^&*()"},
    )
    assert response.status_code == 200
    assert "task_id" in response.json()


def test_classify_long_text_returns_task_id(client_with_queue):
    """POST /classify с длинным текстом возвращает task_id."""
    long_text = "Это очень длинный текст. " * 100
    response = client_with_queue.post(
        "/api/v1/classify",
        json={"text": long_text},
    )
    assert response.status_code == 200
    assert "task_id" in response.json()
