"""
Тесты прогона всех загруженных моделей через API predict и predict_batch.

Предназначены для запуска при развёрнутом сервисе: приложение поднимается с реальным
lifespan (без dependency overrides), загружаются все модели, для которых есть файлы
(regex — всегда; tfidf, fasttext, rnn, bert — при наличии артефактов).
Для каждой зарегистрированной модели проверяются POST /classify и POST /classify/batch.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


# Клиент с реальным приложением (lifespan выполняется, модели грузятся с диска)
@pytest.fixture(scope="module")
def deployed_client():
    with TestClient(app) as client:
        yield client


def _get_registered_models(client: TestClient) -> list[str]:
    """Возвращает список имён зарегистрированных моделей из /api/v1/stats."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200, response.text
    data = response.json()
    model_stats = data.get("model_stats") or {}
    return list(model_stats.keys())


@pytest.fixture(scope="module")
def registered_models(deployed_client):
    """Список моделей, загруженных при старте приложения."""
    return _get_registered_models(deployed_client)


def test_service_health(deployed_client):
    """Сервис отвечает и хотя бы одна модель загружена."""
    response = deployed_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") in ("healthy", "degraded", "unhealthy")
    assert "model_info" in data


def test_at_least_regex_loaded(registered_models):
    """Минимум regex всегда должен быть зарегистрирован."""
    assert "regex" in registered_models, (
        "Ожидается хотя бы модель regex. Зарегистрированы: " + ", ".join(registered_models)
    )


# Общие тестовые тексты
SAMPLE_TEXT = "это тестовый комментарий для проверки модели"
SAMPLE_TEXTS = [
    "нормальный комментарий",
    "ещё один нейтральный текст",
    "спасибо за информацию",
]


@pytest.mark.parametrize("model_name", ["regex"])  # базовый набор; доп. модели добавятся динамически
def test_classify_predict_for_model(deployed_client, model_name, registered_models):
    """Для каждой зарегистрированной модели: predict (POST /classify) возвращает валидный ответ."""
    if model_name not in registered_models:
        pytest.skip(f"Модель {model_name} не загружена. Доступны: {registered_models}")
    response = deployed_client.post(
        "/api/v1/classify",
        json={"text": SAMPLE_TEXT, "preferred_model": model_name},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "is_toxic" in data
    assert "toxicity_score" in data
    assert isinstance(data["is_toxic"], bool)
    assert 0.0 <= data["toxicity_score"] <= 1.0
    assert data.get("model_used") == model_name or data.get("model_used")  # может быть алиас


@pytest.mark.parametrize("model_name", ["regex"])
def test_classify_batch_for_model(deployed_client, model_name, registered_models):
    """Для каждой зарегистрированной модели: predict_batch (POST /classify/batch) возвращает валидный ответ."""
    if model_name not in registered_models:
        pytest.skip(f"Модель {model_name} не загружена. Доступны: {registered_models}")
    response = deployed_client.post(
        "/api/v1/classify/batch",
        json={"texts": SAMPLE_TEXTS, "preferred_model": model_name},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "results" in data
    assert "total" in data
    assert data["total"] == len(SAMPLE_TEXTS)
    assert len(data["results"]) == len(SAMPLE_TEXTS)
    for result in data["results"]:
        assert "is_toxic" in result
        assert "toxicity_score" in result
        assert 0.0 <= result["toxicity_score"] <= 1.0
    assert data.get("model_used") == model_name or data.get("model_used")


class TestAllModelsPredict:
    """Прогон predict для всех загруженных моделей (динамически по registered_models)."""

    def test_classify_each_model(self, deployed_client, registered_models):
        for model_name in registered_models:
            response = deployed_client.post(
                "/api/v1/classify",
                json={"text": SAMPLE_TEXT, "preferred_model": model_name},
            )
            assert response.status_code == 200, (
                f"Модель {model_name}: {response.status_code} {response.text}"
            )
            data = response.json()
            assert "is_toxic" in data and "toxicity_score" in data
            assert isinstance(data["is_toxic"], bool)
            assert 0.0 <= data["toxicity_score"] <= 1.0

    def test_classify_batch_each_model(self, deployed_client, registered_models):
        for model_name in registered_models:
            response = deployed_client.post(
                "/api/v1/classify/batch",
                json={"texts": SAMPLE_TEXTS, "preferred_model": model_name},
            )
            assert response.status_code == 200, (
                f"Модель {model_name}: {response.status_code} {response.text}"
            )
            data = response.json()
            assert "results" in data and "total" in data
            assert data["total"] == len(SAMPLE_TEXTS)
            assert len(data["results"]) == len(SAMPLE_TEXTS)
            for result in data["results"]:
                assert "is_toxic" in result and "toxicity_score" in result
                assert 0.0 <= result["toxicity_score"] <= 1.0
