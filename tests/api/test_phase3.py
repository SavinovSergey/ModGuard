"""Тесты Фазы 3: параллельная токсичность+спам, поля is_spam/spam_score в ответе, кэш."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.model_manager import ModelManager
from app.core.cache import NoOpModerationCache
from app.services.classification import ClassificationService
from app.models.toxicity.regex_model import RegexModel
from app.api.routes import get_classification_service, get_model_manager, get_task_store
from app.core.task_store import TaskStore


@pytest.fixture
def client_phase3():
    """Клиент с ClassificationService (токсичность + спам, без реальной модели спама)."""
    test_model_manager = ModelManager()
    regex_model = RegexModel()
    regex_model.load()
    test_model_manager.register_model("regex", regex_model)
    test_model_manager.set_current_model("regex")
    cache = NoOpModerationCache()
    test_classification_service = ClassificationService(
        test_model_manager,
        moderation_cache=cache,
        spam_model=None,
    )
    store = TaskStore(None)

    app.dependency_overrides[get_classification_service] = lambda: test_classification_service
    app.dependency_overrides[get_model_manager] = lambda: test_model_manager
    app.dependency_overrides[get_task_store] = lambda: store

    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_classify_response_has_spam_fields(client_phase3: TestClient):
    """Ответ /classify содержит is_spam и spam_score."""
    response = client_phase3.post(
        "/api/v1/classify",
        json={"text": "нормальный комментарий"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data
    assert "is_spam" in data
    assert "spam_score" in data
    assert data["is_spam"] is False
    assert data["spam_score"] == 0.0


def test_classify_batch_response_has_spam_fields(client_phase3: TestClient):
    """Ответ /classify/batch содержит is_spam и spam_score в каждом элементе."""
    response = client_phase3.post(
        "/api/v1/classify/batch",
        json={"texts": ["текст один", "ебать второй"]},
    )
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
    service = ClassificationService(model_manager, moderation_cache=None, spam_model=None)
    result = service.classify("ебать тест")
    assert result["is_toxic"] is True
    assert result["is_spam"] is False
    assert result["spam_score"] == 0.0
    batch = service.classify_batch(["норм", "тоже норм"])
    assert len(batch) == 2
    for r in batch:
        assert "is_spam" in r and "spam_score" in r
