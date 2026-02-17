"""Тесты для BERT модели"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.models.bert_model import BERTModel


@pytest.fixture
def bert_model():
    """Загружает реальную BERT модель из models/bert/ для тестирования"""
    model_path = Path("models/bert")
    
    # Проверяем существование модели
    if not model_path.exists():
        pytest.skip(f"Модель не найдена в {model_path}. Обучите модель перед запуском тестов.")
    
    # Проверяем наличие необходимых файлов
    if not (model_path / "config.json").exists():
        pytest.skip(f"Модель в {model_path} неполная. Обучите модель перед запуском тестов.")
    
    model = BERTModel(model_path=str(model_path))
    model.load()
    
    yield model


def test_bert_model_initialization():
    """Тест инициализации модели"""
    model = BERTModel()
    assert model.model_name == "bert"
    assert not model.is_loaded
    assert model.model is None
    assert model.tokenizer is None
    assert model.optimal_threshold == 0.5
    assert model.max_length == 512


def test_bert_model_initialization_with_path():
    """Тест инициализации модели с путем"""
    model = BERTModel(model_path="models/bert")
    assert model.model_name == "bert"
    assert model.model_path == "models/bert"
    assert not model.is_loaded


def test_bert_model_initialization_with_model_name():
    """Тест инициализации модели с именем модели"""
    model = BERTModel(model_name="cointegrated/rubert-tiny2")
    # model_name это тип модели (из BaseToxicityModel), hf_model_name хранит имя из HuggingFace
    assert model.model_name == "bert"
    assert model.hf_model_name == "cointegrated/rubert-tiny2"
    assert not model.is_loaded


def test_bert_model_load_without_path_or_name():
    """Тест загрузки модели без указания пути или имени"""
    model = BERTModel()
    
    with pytest.raises(ValueError, match="Необходимо указать либо model_path"):
        model.load()


def test_bert_model_load_nonexistent_path():
    """Тест загрузки модели с несуществующим путем"""
    model = BERTModel()
    
    with pytest.raises(FileNotFoundError):
        model.load(model_path="nonexistent/model")


def test_bert_model_load_with_params(bert_model):
    """Тест загрузки модели с params.json"""
    # Модель уже загружена в фикстуре
    assert bert_model.is_loaded
    assert bert_model.optimal_threshold is not None
    assert bert_model.max_length is not None


def test_bert_model_predict_not_loaded():
    """Тест предсказания без загрузки модели"""
    model = BERTModel()
    
    with pytest.raises(RuntimeError, match="Модель не загружена"):
        model.predict("тестовый текст")


def test_bert_model_predict_empty_text(bert_model):
    """Тест предсказания для пустого текста"""
    result = bert_model.predict("")
    
    assert result['is_toxic'] is False
    assert result['toxicity_score'] == 0.0
    assert result['toxicity_types'] == {}


def test_bert_model_predict_whitespace(bert_model):
    """Тест предсказания для текста только с пробелами"""
    result = bert_model.predict("   ")
    
    assert result['is_toxic'] is False
    assert result['toxicity_score'] == 0.0


def test_bert_model_predict_normal_text(bert_model):
    """Тест предсказания для нормального текста"""
    result = bert_model.predict("это нормальный комментарий")
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result
    assert 'toxicity_types' in result
    assert isinstance(result['toxicity_score'], float)
    assert 0.0 <= result['toxicity_score'] <= 1.0


def test_bert_model_predict_toxic_text(bert_model):
    """Тест предсказания для токсичного текста"""
    result = bert_model.predict("иди нахуй")
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result
    assert isinstance(result['toxicity_score'], float)
    assert 0.0 <= result['toxicity_score'] <= 1.0


def test_bert_model_predict_with_optimal_threshold(bert_model):
    """Тест предсказания с использованием optimal_threshold"""
    # Сохраняем оригинальный порог
    original_threshold = bert_model.optimal_threshold
    
    # Устанавливаем высокий порог
    bert_model.optimal_threshold = 0.9
    
    result = bert_model.predict("текст")
    
    # Проверяем, что результат соответствует порогу
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result
    
    # Восстанавливаем оригинальный порог
    bert_model.optimal_threshold = original_threshold


def test_bert_model_predict_batch_not_loaded():
    """Тест batch предсказания без загрузки модели"""
    model = BERTModel()
    
    with pytest.raises(RuntimeError, match="Модель не загружена"):
        model.predict_batch(["текст 1", "текст 2"])


def test_bert_model_predict_batch_empty_list(bert_model):
    """Тест batch предсказания для пустого списка"""
    results = bert_model.predict_batch([])
    
    assert results == []


def test_bert_model_predict_batch(bert_model):
    """Тест batch предсказания"""
    texts = [
        "это нормальный комментарий",
        "иди нахуй",
        "спасибо за информацию"
    ]
    
    results = bert_model.predict_batch(texts)
    
    assert len(results) == 3
    for result in results:
        assert isinstance(result, dict)
        assert 'is_toxic' in result
        assert 'toxicity_score' in result
        assert 'toxicity_types' in result
        assert isinstance(result['toxicity_score'], float)
        assert 0.0 <= result['toxicity_score'] <= 1.0


def test_bert_model_predict_batch_with_empty_texts(bert_model):
    """Тест batch предсказания с пустыми текстами"""
    texts = [
        "нормальный текст",
        "",
        "   ",
        "еще один текст"
    ]
    
    results = bert_model.predict_batch(texts)
    
    assert len(results) == 4
    # Пустые тексты должны возвращать нулевую токсичность
    assert results[1]['toxicity_score'] == 0.0
    assert results[1]['is_toxic'] is False
    assert results[2]['toxicity_score'] == 0.0
    assert results[2]['is_toxic'] is False
    # Валидные тексты должны иметь результаты
    assert results[0]['toxicity_score'] > 0.0
    assert results[3]['toxicity_score'] > 0.0


def test_bert_model_predict_batch_all_empty(bert_model):
    """Тест batch предсказания когда все тексты пустые"""
    texts = ["", "   ", ""]
    
    results = bert_model.predict_batch(texts)
    
    assert len(results) == 3
    for result in results:
        assert result['toxicity_score'] == 0.0
        assert result['is_toxic'] is False


def test_bert_model_get_model_info_not_loaded():
    """Тест получения информации о модели без загрузки"""
    model = BERTModel()
    info = model.get_model_info()
    
    assert info['name'] == 'bert'
    assert info['type'] == 'bert'
    assert info['is_loaded'] is False
    assert info['version'] == '1.0'
    assert 'description' in info


def test_bert_model_get_model_info_loaded(bert_model):
    """Тест получения информации о загруженной модели"""
    info = bert_model.get_model_info()
    
    assert info['type'] == 'bert'
    assert info['is_loaded'] is True
    assert 'optimal_threshold' in info
    assert 'max_length' in info
    assert 'device' in info


def test_bert_model_get_model_info_with_path():
    """Тест получения информации о модели с путем"""
    model = BERTModel(model_path="models/bert")
    info = model.get_model_info()
    
    assert info['name'] == 'models/bert' or info['name'] == 'bert'  # Может быть путь или 'bert' если путь None


def test_bert_model_get_model_info_with_model_name():
    """Тест получения информации о модели с именем модели"""
    model = BERTModel(model_name="cointegrated/rubert-tiny2")
    info = model.get_model_info()
    
    # hf_model_name хранится как имя модели из HuggingFace
    assert info['name'] == 'cointegrated/rubert-tiny2'


def test_bert_model_load_from_huggingface():
    """Тест загрузки модели из HuggingFace по имени"""
    with patch('app.models.bert_model.AutoModelForSequenceClassification') as mock_model_class, \
         patch('app.models.bert_model.AutoTokenizer') as mock_tokenizer_class:
        
        import torch
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        model = BERTModel(model_name="cointegrated/rubert-tiny2")
        model.load()
        
        assert model.is_loaded
        # Проверяем, что from_pretrained был вызван с правильным именем
        mock_model_class.from_pretrained.assert_called_once_with("cointegrated/rubert-tiny2")
        mock_tokenizer_class.from_pretrained.assert_called_once_with("cointegrated/rubert-tiny2")


def test_bert_model_text_preprocessing(bert_model):
    """Тест предобработки текста"""
    # Проверяем, что используется TextProcessor
    assert bert_model.text_processor is not None
    assert hasattr(bert_model.text_processor, 'normalize')


def test_bert_model_device_setting():
    """Тест настройки устройства"""
    model = BERTModel()
    # Устройство должно быть установлено автоматически
    assert model.device in ['cuda', 'cpu']


def test_bert_model_max_length_setting():
    """Тест настройки max_length"""
    model = BERTModel(max_length=256)
    assert model.max_length == 256
    
    model2 = BERTModel()
    assert model2.max_length == 512  # Значение по умолчанию

