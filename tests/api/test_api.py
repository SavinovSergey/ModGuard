"""Интеграционные тесты для API endpoints"""
import pytest
import pickle
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.main import app
from app.core.model_manager import ModelManager
from app.services.classification import ClassificationService
from app.models.regex_model import RegexModel
from app.models.tfidf_model import TfidfModel
from app.preprocessing.text_processor import TextProcessor


@pytest.fixture
def temp_tfidf_model_files():
    """Создает временные файлы TF-IDF модели для тестирования"""
    text_processor = TextProcessor()
    
    # Простые тестовые данные
    texts = [
        "это нормальный комментарий",
        "это токсичный ебать комментарий",
        "спасибо за информацию",
        "иди нахуй",
        "интересная статья"
    ]
    labels = [0, 1, 0, 1, 0]  # 0 = нетоксичный, 1 = токсичный
    
    # Предобработка текстов
    processed_texts = [text_processor.process(text) for text in texts]
    
    # Создаем простую модель
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed_texts)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, labels)
    
    # Сохраняем во временные файлы
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / 'model.pkl'
        vectorizer_path = Path(tmpdir) / 'vectorizer.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        yield str(model_path), str(vectorizer_path)


@pytest.fixture
def client_with_regex():
    """Создает тестовый клиент с только regex моделью"""
    # Инициализация менеджера моделей
    test_model_manager = ModelManager()
    
    # Регистрация только regex модели
    regex_model = RegexModel()
    regex_model.load()
    test_model_manager.register_model("regex", regex_model)
    test_model_manager.current_model = regex_model
    
    # Инициализация сервиса классификации
    test_classification_service = ClassificationService(test_model_manager)
    
    # Переопределяем зависимости
    from app.api.routes import get_classification_service, get_model_manager
    
    def override_get_classification_service():
        return test_classification_service
    
    def override_get_model_manager():
        return test_model_manager
    
    app.dependency_overrides[get_classification_service] = override_get_classification_service
    app.dependency_overrides[get_model_manager] = override_get_model_manager
    
    client = TestClient(app)
    
    yield client
    
    # Очищаем переопределения
    app.dependency_overrides.clear()


@pytest.fixture
def client_with_tfidf(temp_tfidf_model_files):
    """Создает тестовый клиент с TF-IDF моделью"""
    model_path, vectorizer_path = temp_tfidf_model_files
    
    # Инициализация менеджера моделей
    test_model_manager = ModelManager()
    
    # Регистрация regex модели
    regex_model = RegexModel()
    regex_model.load()
    test_model_manager.register_model("regex", regex_model)
    
    # Регистрация TF-IDF модели
    tfidf_model = TfidfModel()
    tfidf_model.load(model_path=model_path, vectorizer_path=vectorizer_path)
    test_model_manager.register_model("tfidf", tfidf_model)
    test_model_manager.current_model = tfidf_model
    
    # Инициализация сервиса классификации
    test_classification_service = ClassificationService(test_model_manager)
    
    # Переопределяем зависимости
    from app.api.routes import get_classification_service, get_model_manager
    
    def override_get_classification_service():
        return test_classification_service
    
    def override_get_model_manager():
        return test_model_manager
    
    app.dependency_overrides[get_classification_service] = override_get_classification_service
    app.dependency_overrides[get_model_manager] = override_get_model_manager
    
    client = TestClient(app)
    
    yield client
    
    # Очищаем переопределения
    app.dependency_overrides.clear()


def test_root_endpoint(client_with_regex):
    """Тест корневого endpoint"""
    response = client_with_regex.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ToxicFilter"
    assert data["status"] == "running"
    assert "version" in data


def test_classify_endpoint_basic(client_with_regex):
    """Тест базового endpoint классификации"""
    response = client_with_regex.post(
        "/api/v1/classify",
        json={"text": "это нормальный комментарий"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data
    assert "toxicity_score" in data
    assert "toxicity_types" in data
    assert isinstance(data["is_toxic"], bool)
    assert 0.0 <= data["toxicity_score"] <= 1.0


def test_classify_endpoint_toxic_text(client_with_regex):
    """Тест классификации токсичного текста"""
    response = client_with_regex.post(
        "/api/v1/classify",
        json={"text": "иди нахуй ебать"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["is_toxic"] is True
    assert data["toxicity_score"] > 0.0
    assert "model_used" in data


def test_classify_endpoint_non_toxic_text(client_with_regex):
    """Тест классификации нетоксичного текста"""
    response = client_with_regex.post(
        "/api/v1/classify",
        json={"text": "спасибо за информацию, очень полезно"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["is_toxic"] is False
    assert "model_used" in data


def test_classify_endpoint_with_preferred_model(client_with_regex):
    """Тест классификации с указанием предпочтительной модели"""
    response = client_with_regex.post(
        "/api/v1/classify",
        json={
            "text": "тестовый текст",
            "preferred_model": "regex"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data
    assert data.get("model_used") == "regex"


def test_classify_endpoint_with_context(client_with_regex):
    """Тест классификации с контекстом"""
    response = client_with_regex.post(
        "/api/v1/classify",
        json={
            "text": "это ответ на предыдущее сообщение",
            "context": ["предыдущее сообщение", "еще одно"]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data


def test_classify_endpoint_empty_text(client_with_regex):
    """Тест классификации пустого текста"""
    response = client_with_regex.post(
        "/api/v1/classify",
        json={"text": ""}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["is_toxic"] is False
    assert data["toxicity_score"] == 0.0


def test_classify_batch_endpoint(client_with_regex):
    """Тест batch endpoint классификации"""
    response = client_with_regex.post(
        "/api/v1/classify/batch",
        json={
            "texts": [
                "нормальный комментарий",
                "токсичный ебать комментарий",
                "спасибо за помощь"
            ]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total" in data
    assert len(data["results"]) == 3
    assert data["total"] == 3
    
    for result in data["results"]:
        assert "is_toxic" in result
        assert "toxicity_score" in result
        assert 0.0 <= result["toxicity_score"] <= 1.0


def test_classify_batch_endpoint_with_preferred_model(client_with_regex):
    """Тест batch endpoint с указанием модели"""
    response = client_with_regex.post(
        "/api/v1/classify/batch",
        json={
            "texts": ["текст 1", "текст 2"],
            "preferred_model": "regex"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data.get("model_used") == "regex"
    assert len(data["results"]) == 2


def test_classify_batch_endpoint_empty_list(client_with_regex):
    """Тест batch endpoint с пустым списком"""
    response = client_with_regex.post(
        "/api/v1/classify/batch",
        json={"texts": []}
    )
    
    # Должна быть ошибка валидации
    assert response.status_code == 422


def test_classify_batch_endpoint_too_large(client_with_regex):
    """Тест batch endpoint с превышением лимита"""
    large_texts = [f"текст {i}" for i in range(1001)]
    
    response = client_with_regex.post(
        "/api/v1/classify/batch",
        json={"texts": large_texts}
    )
    
    assert response.status_code == 400
    assert "exceeds maximum" in response.json()["detail"].lower()


def test_health_endpoint(client_with_regex):
    """Тест health check endpoint"""
    response = client_with_regex.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_info" in data
    assert data["status"] in ["healthy", "degraded", "unhealthy"]


def test_health_endpoint_model_info(client_with_regex):
    """Тест что health endpoint возвращает информацию о модели"""
    response = client_with_regex.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    model_info = data["model_info"]
    assert "name" in model_info
    assert "is_loaded" in model_info


def test_stats_endpoint(client_with_regex):
    """Тест stats endpoint"""
    response = client_with_regex.get("/api/v1/stats")
    
    assert response.status_code == 200
    data = response.json()
    assert "model_stats" in data
    assert "current_model" in data
    assert isinstance(data["model_stats"], dict)


def test_classify_with_tfidf_model(client_with_tfidf):
    """Тест классификации с TF-IDF моделью"""
    response = client_with_tfidf.post(
        "/api/v1/classify",
        json={
            "text": "тестовый текст для классификации",
            "preferred_model": "tfidf"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data
    assert "toxicity_score" in data
    assert data.get("model_used") == "tfidf"


def test_classify_batch_with_tfidf_model(client_with_tfidf):
    """Тест batch классификации с TF-IDF моделью"""
    response = client_with_tfidf.post(
        "/api/v1/classify/batch",
        json={
            "texts": [
                "нормальный комментарий",
                "токсичный комментарий"
            ],
            "preferred_model": "tfidf"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data.get("model_used") == "tfidf"
    
    for result in data["results"]:
        assert "is_toxic" in result
        assert "toxicity_score" in result


def test_classify_fallback_to_regex(client_with_tfidf):
    """Тест fallback на regex модель если TF-IDF недоступна"""
    # Запрашиваем несуществующую модель, должна использоваться regex
    response = client_with_tfidf.post(
        "/api/v1/classify",
        json={
            "text": "тестовый текст",
            "preferred_model": "nonexistent_model"
        }
    )
    
    # Должен вернуться успешный ответ с fallback моделью
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data
    # Модель должна быть либо regex, либо tfidf (из fallback chain)


def test_classify_invalid_json(client_with_regex):
    """Тест обработки невалидного JSON"""
    response = client_with_regex.post(
        "/api/v1/classify",
        json={"invalid": "data"}
    )
    
    # Должна быть ошибка валидации
    assert response.status_code == 422


def test_classify_missing_text_field(client_with_regex):
    """Тест запроса без обязательного поля text"""
    response = client_with_regex.post(
        "/api/v1/classify",
        json={}
    )
    
    assert response.status_code == 422


def test_multiple_classify_requests(client_with_regex):
    """Тест множественных запросов для проверки стабильности"""
    texts = [
        "нормальный комментарий",
        "токсичный ебать",
        "спасибо",
        "иди нахуй"
    ]
    
    for text in texts:
        response = client_with_regex.post(
            "/api/v1/classify",
            json={"text": text}
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_toxic" in data
        assert "toxicity_score" in data


def test_classify_special_characters(client_with_regex):
    """Тест обработки специальных символов"""
    response = client_with_regex.post(
        "/api/v1/classify",
        json={"text": "Текст с эмодзи 😀 и спецсимволами !@#$%^&*()"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data


def test_classify_long_text(client_with_regex):
    """Тест обработки длинного текста"""
    long_text = "Это очень длинный текст. " * 100
    
    response = client_with_regex.post(
        "/api/v1/classify",
        json={"text": long_text}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "is_toxic" in data

