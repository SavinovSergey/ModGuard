"""Тесты для FastText модели"""
import pytest
import tempfile
from pathlib import Path
import numpy as np
import fasttext

from app.models.toxicity.fasttext_model import FastTextModel
from app.preprocessing.text_processor import TextProcessor


@pytest.fixture
def temp_model_file():
    """Создает временный файл FastText модели для тестирования"""
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
    
    # Создаем временный файл для обучения FastText
    with tempfile.TemporaryDirectory() as tmpdir:
        # Сохраняем тексты в файл для FastText в формате train_supervised
        fasttext_data_path = Path(tmpdir) / 'fasttext_data.txt'
        with open(fasttext_data_path, 'w', encoding='utf-8') as f:
            for text, label in zip(processed_texts, labels):
                label_str = '__label__1' if label == 1 else '__label__0'
                f.write(f"{label_str} {text}\n")
        
        # Обучаем FastText модель через train_supervised
        fasttext_model = fasttext.train_supervised(
            str(fasttext_data_path),
            dim=100,  # Меньшая размерность для быстрых тестов
            epoch=5,
            lr=0.1,
            wordNgrams=1,
            minCount=1,
            verbose=0
        )
        
        # Сохраняем модель
        model_path = Path(tmpdir) / 'fasttext_model.bin'
        fasttext_model.save_model(str(model_path))
        
        yield str(model_path)


@pytest.fixture
def fasttext_model(temp_model_file):
    """Создает и загружает FastText модель для тестирования"""
    model = FastTextModel()
    model.load(model_path=temp_model_file)
    return model


def test_fasttext_model_initialization():
    """Тест инициализации модели"""
    model = FastTextModel()
    assert model.model_name == "fasttext"
    assert not model.is_loaded
    assert model.model is None
    assert model.model_path is None


def test_fasttext_model_initialization_with_path():
    """Тест инициализации модели с путем"""
    model = FastTextModel(model_path="models/toxicity/fasttext/fasttext_model.bin")
    assert model.model_name == "fasttext"
    assert model.model_path == "models/toxicity/fasttext/fasttext_model.bin"
    # is_loaded будет True только если файл существует и загружен
    # В данном случае файл не существует, поэтому False
    assert not model.is_loaded


def test_fasttext_model_load(temp_model_file):
    """Тест загрузки модели"""
    model = FastTextModel()
    
    model.load(model_path=temp_model_file)
    
    assert model.is_loaded
    assert model.model is not None
    # Проверяем что это FastText модель (используем hasattr вместо isinstance)
    assert hasattr(model.model, 'predict')
    assert model.model_path == temp_model_file


def test_fasttext_model_load_without_path():
    """Тест загрузки модели без указания пути"""
    model = FastTextModel()
    
    with pytest.raises(ValueError, match="Необходимо указать путь"):
        model.load()


def test_fasttext_model_load_nonexistent_file():
    """Тест загрузки модели с несуществующим файлом"""
    model = FastTextModel()
    
    with pytest.raises(FileNotFoundError):
        model.load(model_path="nonexistent/fasttext_model.bin")


def test_fasttext_model_predict_not_loaded():
    """Тест предсказания без загрузки модели"""
    model = FastTextModel()
    
    with pytest.raises(RuntimeError, match="Модель не загружена"):
        model.predict("тестовый текст")


def test_fasttext_model_predict_empty_text(fasttext_model):
    """Тест предсказания для пустого текста"""
    result = fasttext_model.predict("")
    
    assert result['is_toxic'] is False
    # Пустые тексты должны возвращать нулевую вероятность
    assert result['toxicity_score'] == 0.0
    assert result['toxicity_types'] == {}


def test_fasttext_model_predict_whitespace(fasttext_model):
    """Тест предсказания для текста только с пробелами"""
    result = fasttext_model.predict("   ")
    
    assert result['is_toxic'] is False
    # Тексты только с пробелами должны возвращать нулевую вероятность
    assert result['toxicity_score'] == 0.0


def test_fasttext_model_predict_normal_text(fasttext_model):
    """Тест предсказания для нормального текста"""
    result = fasttext_model.predict("это нормальный комментарий")
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result
    assert 'toxicity_types' in result
    assert isinstance(result['toxicity_score'], float)
    assert 0.0 <= result['toxicity_score'] <= 1.0


def test_fasttext_model_predict_toxic_text(fasttext_model):
    """Тест предсказания для токсичного текста"""
    result = fasttext_model.predict("иди нахуй")
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result
    assert isinstance(result['toxicity_score'], float)
    assert 0.0 <= result['toxicity_score'] <= 1.0


def test_fasttext_model_predict_batch_not_loaded():
    """Тест batch предсказания без загрузки модели"""
    model = FastTextModel()
    
    with pytest.raises(RuntimeError, match="Модель не загружена"):
        model.predict_batch(["текст 1", "текст 2"])


def test_fasttext_model_predict_batch_empty_list(fasttext_model):
    """Тест batch предсказания для пустого списка"""
    results = fasttext_model.predict_batch([])
    
    assert results == []


def test_fasttext_model_predict_batch(fasttext_model):
    """Тест batch предсказания"""
    texts = [
        "это нормальный комментарий",
        "иди нахуй",
        "спасибо за информацию"
    ]
    
    results = fasttext_model.predict_batch(texts)
    
    assert len(results) == 3
    for result in results:
        assert isinstance(result, dict)
        assert 'is_toxic' in result
        assert 'toxicity_score' in result
        assert 'toxicity_types' in result
        assert isinstance(result['toxicity_score'], float)
        assert 0.0 <= result['toxicity_score'] <= 1.0


def test_fasttext_model_predict_batch_with_empty_texts(fasttext_model):
    """Тест batch предсказания с пустыми текстами"""
    texts = ["", "   ", "нормальный текст"]
    
    results = fasttext_model.predict_batch(texts)
    
    assert len(results) == 3
    # Пустые тексты должны возвращать is_toxic=False
    assert results[0]['is_toxic'] is False
    assert results[1]['is_toxic'] is False


def test_fasttext_model_predict_batch_all_empty(fasttext_model):
    """Тест batch предсказания когда все тексты пустые"""
    texts = ["", "   ", "  "]
    
    results = fasttext_model.predict_batch(texts)
    
    assert len(results) == 3
    for result in results:
        assert result['is_toxic'] is False
        # Пустые тексты должны возвращать нулевую вероятность
        assert result['toxicity_score'] == 0.0


def test_fasttext_model_info_not_loaded():
    """Тест получения информации о незагруженной модели"""
    model = FastTextModel()
    info = model.get_model_info()
    
    assert info['name'] == 'fasttext'
    assert info['type'] == 'fasttext'
    assert info['is_loaded'] is False
    assert 'version' in info
    assert 'description' in info
    assert 'model_params' not in info


def test_fasttext_model_info_loaded(fasttext_model):
    """Тест получения информации о загруженной модели"""
    info = fasttext_model.get_model_info()
    
    assert info['name'] == 'fasttext'
    assert info['type'] == 'fasttext'
    assert info['is_loaded'] is True
    assert 'version' in info
    assert 'description' in info
    assert 'model_params' in info
    # Проверяем наличие основных параметров
    assert 'dim' in info['model_params'] or info['model_params'].get('dim') is not None


def test_fasttext_model_predict_consistency(fasttext_model):
    """Тест консистентности предсказаний"""
    text = "тестовый текст для проверки"
    
    result1 = fasttext_model.predict(text)
    result2 = fasttext_model.predict(text)
    
    # Предсказания должны быть одинаковыми
    assert result1['is_toxic'] == result2['is_toxic']
    assert abs(result1['toxicity_score'] - result2['toxicity_score']) < 1e-6


def test_fasttext_model_text_preprocessing(fasttext_model):
    """Тест что модель правильно обрабатывает текст"""
    # Текст с разными регистрами и знаками препинания
    text = "Это ТЕКСТ с разными РЕГИСТРАМИ!!! И знаками препинания..."
    
    result = fasttext_model.predict(text)
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result


def test_fasttext_model_load_saves_paths(temp_model_file):
    """Тест что путь сохраняется после загрузки"""
    model = FastTextModel()
    
    model.load(model_path=temp_model_file)
    
    # Путь должен быть сохранен
    assert model.model_path == temp_model_file


def test_fasttext_model_load_with_instance_path(temp_model_file):
    """Тест загрузки модели с путем, указанным при инициализации"""
    model = FastTextModel(model_path=temp_model_file)
    
    # Загрузка без параметров должна использовать сохраненный путь
    model.load()
    
    assert model.is_loaded
    assert model.model is not None
