"""Тесты для TF-IDF модели"""
import pytest
import pickle
import tempfile
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.models.toxicity.tfidf_model import TfidfModel
from app.preprocessing.text_processor import TextProcessor


@pytest.fixture
def temp_model_files():
    """Создает временные файлы модели и векторизатора для тестирования"""
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
def tfidf_model(temp_model_files):
    """Создает и загружает TF-IDF модель для тестирования"""
    model_path, vectorizer_path = temp_model_files
    model = TfidfModel()
    model.load(model_path=model_path, vectorizer_path=vectorizer_path)
    return model


def test_tfidf_model_initialization():
    """Тест инициализации модели"""
    model = TfidfModel()
    assert model.model_name == "tfidf"
    assert not model.is_loaded
    assert model.model is None
    assert model.vectorizer is None


def test_tfidf_model_initialization_with_paths():
    """Тест инициализации модели с путями"""
    model = TfidfModel(
        model_path="models/toxicity/tfidf/model.pkl",
        vectorizer_path="models/toxicity/tfidf/vectorizer.pkl"
    )
    assert model.model_name == "tfidf"
    assert model.model_path == "models/toxicity/tfidf/model.pkl"
    assert model.vectorizer_path == "models/toxicity/tfidf/vectorizer.pkl"
    # is_loaded будет True только если файлы существуют и загружены
    # В данном случае файлы не существуют, поэтому False
    assert not model.is_loaded


def test_tfidf_model_load(temp_model_files):
    """Тест загрузки модели"""
    model_path, vectorizer_path = temp_model_files
    model = TfidfModel()
    
    model.load(model_path=model_path, vectorizer_path=vectorizer_path)
    
    assert model.is_loaded
    assert model.model is not None
    assert model.vectorizer is not None
    assert isinstance(model.model, LogisticRegression)
    assert isinstance(model.vectorizer, TfidfVectorizer)


def test_tfidf_model_load_without_paths():
    """Тест загрузки модели без указания путей"""
    model = TfidfModel()
    
    with pytest.raises(ValueError, match="Необходимо указать пути"):
        model.load()


def test_tfidf_model_load_nonexistent_file():
    """Тест загрузки модели с несуществующим файлом"""
    model = TfidfModel()
    
    with pytest.raises(FileNotFoundError):
        model.load(
            model_path="nonexistent/model.pkl",
            vectorizer_path="nonexistent/vectorizer.pkl"
        )


def test_tfidf_model_predict_not_loaded():
    """Тест предсказания без загрузки модели"""
    model = TfidfModel()
    
    with pytest.raises(RuntimeError, match="Модель не загружена"):
        model.predict("тестовый текст")


def test_tfidf_model_predict_empty_text(tfidf_model):
    """Тест предсказания для пустого текста"""
    result = tfidf_model.predict("")
    
    assert result['is_toxic'] is False
    assert result['toxicity_score'] == 0.0
    assert result['toxicity_types'] == {}


def test_tfidf_model_predict_whitespace(tfidf_model):
    """Тест предсказания для текста только с пробелами"""
    result = tfidf_model.predict("   ")
    
    assert result['is_toxic'] is False
    assert result['toxicity_score'] == 0.0


def test_tfidf_model_predict_normal_text(tfidf_model):
    """Тест предсказания для нормального текста"""
    result = tfidf_model.predict("это нормальный комментарий")
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result
    assert 'toxicity_types' in result
    assert isinstance(result['toxicity_score'], float)
    assert 0.0 <= result['toxicity_score'] <= 1.0


def test_tfidf_model_predict_toxic_text(tfidf_model):
    """Тест предсказания для токсичного текста"""
    result = tfidf_model.predict("иди нахуй")
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result
    assert isinstance(result['toxicity_score'], float)
    assert 0.0 <= result['toxicity_score'] <= 1.0


def test_tfidf_model_predict_batch_not_loaded():
    """Тест batch предсказания без загрузки модели"""
    model = TfidfModel()
    
    with pytest.raises(RuntimeError, match="Модель не загружена"):
        model.predict_batch(["текст 1", "текст 2"])


def test_tfidf_model_predict_batch_empty_list(tfidf_model):
    """Тест batch предсказания для пустого списка"""
    results = tfidf_model.predict_batch([])
    
    assert results == []


def test_tfidf_model_predict_batch(tfidf_model):
    """Тест batch предсказания"""
    texts = [
        "это нормальный комментарий",
        "иди нахуй",
        "спасибо за информацию"
    ]
    
    results = tfidf_model.predict_batch(texts)
    
    assert len(results) == 3
    for result in results:
        assert isinstance(result, dict)
        assert 'is_toxic' in result
        assert 'toxicity_score' in result
        assert 'toxicity_types' in result
        assert isinstance(result['toxicity_score'], float)
        assert 0.0 <= result['toxicity_score'] <= 1.0


def test_tfidf_model_predict_batch_with_empty_texts(tfidf_model):
    """Тест batch предсказания с пустыми текстами"""
    texts = ["", "   ", "нормальный текст"]
    
    results = tfidf_model.predict_batch(texts)
    
    assert len(results) == 3
    # Пустые тексты должны возвращать is_toxic=False
    assert results[0]['is_toxic'] is False
    assert results[1]['is_toxic'] is False


def test_tfidf_model_predict_batch_all_empty(tfidf_model):
    """Тест batch предсказания когда все тексты пустые"""
    texts = ["", "   ", "  "]
    
    results = tfidf_model.predict_batch(texts)
    
    assert len(results) == 3
    for result in results:
        assert result['is_toxic'] is False
        assert result['toxicity_score'] == 0.0


def test_tfidf_model_info_not_loaded():
    """Тест получения информации о незагруженной модели"""
    model = TfidfModel()
    info = model.get_model_info()
    
    assert info['name'] == 'tfidf'
    assert info['type'] == 'tfidf'
    assert info['is_loaded'] is False
    assert 'version' in info
    assert 'description' in info
    assert 'model_params' not in info
    assert 'vectorizer_params' not in info


def test_tfidf_model_info_loaded(tfidf_model):
    """Тест получения информации о загруженной модели"""
    info = tfidf_model.get_model_info()
    
    assert info['name'] == 'tfidf'
    assert info['type'] == 'tfidf'
    assert info['is_loaded'] is True
    assert 'version' in info
    assert 'description' in info
    assert 'model_params' in info
    assert 'vectorizer_params' in info
    assert 'C' in info['model_params']
    assert 'max_features' in info['vectorizer_params']


def test_tfidf_model_predict_consistency(tfidf_model):
    """Тест консистентности предсказаний"""
    text = "тестовый текст для проверки"
    
    result1 = tfidf_model.predict(text)
    result2 = tfidf_model.predict(text)
    
    # Предсказания должны быть одинаковыми
    assert result1['is_toxic'] == result2['is_toxic']
    assert abs(result1['toxicity_score'] - result2['toxicity_score']) < 1e-6


def test_tfidf_model_text_preprocessing(tfidf_model):
    """Тест что модель правильно обрабатывает текст"""
    # Текст с разными регистрами и знаками препинания
    text = "Это ТЕКСТ с разными РЕГИСТРАМИ!!! И знаками препинания..."
    
    result = tfidf_model.predict(text)
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result


def test_tfidf_model_load_saves_paths(temp_model_files):
    """Тест что пути сохраняются после загрузки"""
    model_path, vectorizer_path = temp_model_files
    model = TfidfModel()
    
    model.load(model_path=model_path, vectorizer_path=vectorizer_path)
    
    # Пути должны быть сохранены
    assert model.model_path == model_path
    assert model.vectorizer_path == vectorizer_path


def test_tfidf_model_load_with_instance_paths(temp_model_files):
    """Тест загрузки модели с путями, указанными при инициализации"""
    model_path, vectorizer_path = temp_model_files
    model = TfidfModel(
        model_path=model_path,
        vectorizer_path=vectorizer_path
    )
    
    # Загрузка без параметров должна использовать сохраненные пути
    model.load()
    
    assert model.is_loaded
    assert model.model is not None
    assert model.vectorizer is not None

