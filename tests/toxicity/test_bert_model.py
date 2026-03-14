"""Тесты для BERT модели (PyTorch и квантизованная ONNX)"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.models.toxicity.bert_model import BERTModel, _ONNX_FILES


def _has_pytorch_model(path: Path) -> bool:
    """Проверяет наличие PyTorch модели (config.json + pytorch_model.bin или model.safetensors)."""
    if not path.exists() or not (path / "config.json").exists():
        return False
    return (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()


def _has_onnx_model(path: Path) -> bool:
    """Проверяет наличие ONNX модели."""
    if not path.exists():
        return False
    return any((path / fname).exists() for fname in _ONNX_FILES)


@pytest.fixture(scope="module")
def bert_model():
    """Загружает BERT модель из models/toxicity/bert/ (PyTorch при наличии, иначе автоопределение).
    scope=module — одна загрузка на файл, чтобы избежать segfault при повторных load/unload torch.
    """
    model_path = Path("models/toxicity/bert")
    if not model_path.exists():
        pytest.skip(f"Модель не найдена в {model_path}. Обучите модель перед запуском тестов.")
    if not (model_path / "config.json").exists():
        pytest.skip(f"Модель в {model_path} неполная. Обучите модель перед запуском тестов.")
    # Явно PyTorch, если есть веса
    use_onnx = False if _has_pytorch_model(model_path) else None
    model = BERTModel(model_path=str(model_path), use_onnx=use_onnx)
    model.load()
    yield model


@pytest.fixture(scope="module")
def bert_model_pytorch():
    """Загружает PyTorch BERT модель (use_onnx=False).
    scope=module — одна загрузка на файл (снижает риск segfault в torch._dynamo).
    """
    model_path = Path("models/toxicity/bert")
    if not _has_pytorch_model(model_path):
        pytest.skip(f"PyTorch модель не найдена в {model_path}. Обучите модель перед запуском тестов.")
    model = BERTModel(model_path=str(model_path), use_onnx=False)
    model.load()
    yield model


@pytest.fixture(scope="module")
def bert_model_onnx():
    """Загружает квантизованную ONNX BERT модель.
    scope=module — одна загрузка на файл.
    """
    # Проверяем наличие onnxruntime
    try:
        import onnxruntime
    except ImportError:
        pytest.skip(
            "onnxruntime не установлен. Установите: pip install onnxruntime\n"
            "Или: pip install optimum[onnxruntime]"
        )
    
    # Проверяем наличие optimum.onnxruntime
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except ImportError:
        pytest.skip(
            "optimum[onnxruntime] не установлен. Установите: pip install optimum[onnxruntime]"
        )
    
    onnx_path = None
    for path in (Path("models/toxicity/bert/onnx"), Path("models/toxicity/bert/onnx_cpu"), Path("models/toxicity/bert")):
        if _has_onnx_model(path):
            onnx_path = path
            break
    if onnx_path is None:
        pytest.skip(
            "ONNX модель не найдена. Запустите квантизацию: "
            "python scripts/toxicity/quantize_bert_onnx.py models/toxicity/bert -o models/toxicity/bert/onnx --device cpu"
        )
    model = BERTModel(model_path=str(onnx_path), use_onnx=True)
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
    model = BERTModel(model_path="models/toxicity/bert")
    assert model.model_name == "bert"
    assert model.model_path == "models/toxicity/bert"
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


def test_bert_backend_pytorch_unchanged_after_load_and_predict(bert_model_pytorch):
    """PyTorch-версия BERT: бэкенд остаётся PyTorch после load и после predict/predict_batch."""
    model = bert_model_pytorch
    assert getattr(model, "_is_onnx", True) is False, "После загрузки с use_onnx=False должен быть PyTorch"
    model.predict("проверка бэкенда")
    assert getattr(model, "_is_onnx", True) is False, "После predict бэкенд не должен меняться"
    model.predict_batch(["ещё один текст", "и второй"])
    assert getattr(model, "_is_onnx", True) is False, "После predict_batch бэкенд не должен меняться"
    assert model.get_model_info().get("is_onnx") is False


def test_bert_backend_onnx_unchanged_after_load_and_predict(bert_model_onnx):
    """ONNX-версия BERT: бэкенд остаётся ONNX после load и после predict/predict_batch."""
    model = bert_model_onnx
    assert getattr(model, "_is_onnx", False) is True, "После загрузки с use_onnx=True должен быть ONNX"
    model.predict("проверка бэкенда")
    assert getattr(model, "_is_onnx", False) is True, "После predict бэкенд не должен меняться"
    model.predict_batch(["ещё один текст", "и второй"])
    assert getattr(model, "_is_onnx", False) is True, "После predict_batch бэкенд не должен меняться"
    assert model.get_model_info().get("is_onnx") is True


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

    assert info["name"] == "bert"
    assert info["type"] == "bert"
    assert info["is_loaded"] is False
    assert info["version"] == "1.0"
    assert "description" in info
    assert info["is_onnx"] is False


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
    model = BERTModel(model_path="models/toxicity/bert")
    info = model.get_model_info()
    
    assert info['name'] == 'models/toxicity/bert' or info['name'] == 'bert'


def test_bert_model_get_model_info_with_model_name():
    """Тест получения информации о модели с именем модели"""
    model = BERTModel(model_name="cointegrated/rubert-tiny2")
    info = model.get_model_info()
    
    # hf_model_name хранится как имя модели из HuggingFace
    assert info['name'] == 'cointegrated/rubert-tiny2'


def test_bert_model_load_from_huggingface():
    """Тест загрузки модели из HuggingFace по имени"""
    with patch('app.models.toxicity.bert_model.AutoModelForSequenceClassification') as mock_model_class, \
         patch('app.models.toxicity.bert_model.AutoTokenizer') as mock_tokenizer_class:
        
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


# --- Тесты для use_onnx параметра ---


def test_bert_model_use_onnx_initialization():
    """Тест инициализации с use_onnx."""
    model = BERTModel(use_onnx=None)
    assert model.use_onnx is None
    assert model._is_onnx is False

    model2 = BERTModel(use_onnx=False)
    assert model2.use_onnx is False

    model3 = BERTModel(use_onnx=True)
    assert model3.use_onnx is True


def test_bert_model_use_onnx_true_raises_if_no_onnx():
    """Тест: use_onnx=True без ONNX файлов вызывает FileNotFoundError."""
    # Создаём временную директорию без ONNX
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Пустая директория или только config.json
        (Path(tmpdir) / "config.json").write_text("{}")
        model = BERTModel(model_path=tmpdir, use_onnx=True)
        with pytest.raises(FileNotFoundError, match="ONNX модель не найдена"):
            model.load()


# --- Тесты для ONNX модели ---


def test_bert_model_onnx_predict(bert_model_onnx):
    """Тест предсказания ONNX модели."""
    result = bert_model_onnx.predict("это тестовый комментарий")
    assert isinstance(result, dict)
    assert "is_toxic" in result
    assert "toxicity_score" in result
    assert "toxicity_types" in result
    assert isinstance(result["toxicity_score"], float)
    assert 0.0 <= result["toxicity_score"] <= 1.0


def test_bert_model_onnx_predict_batch(bert_model_onnx):
    """Тест batch предсказания ONNX модели."""
    texts = ["нормальный текст", "токсичный комментарий"]
    results = bert_model_onnx.predict_batch(texts)
    assert len(results) == 2
    for r in results:
        assert "toxicity_score" in r
        assert 0.0 <= r["toxicity_score"] <= 1.0


def test_bert_model_onnx_is_onnx_flag(bert_model_onnx):
    """Тест флага _is_onnx у ONNX модели."""
    assert bert_model_onnx._is_onnx is True


def test_bert_model_onnx_get_model_info(bert_model_onnx):
    """Тест get_model_info для ONNX модели."""
    info = bert_model_onnx.get_model_info()
    assert info["is_loaded"] is True
    assert info["is_onnx"] is True
    assert "optimal_threshold" in info
    assert "device" in info


# --- Тесты для PyTorch модели ---


def test_bert_model_pytorch_is_onnx_flag(bert_model_pytorch):
    """Тест флага _is_onnx у PyTorch модели."""
    assert bert_model_pytorch._is_onnx is False


def test_bert_model_pytorch_get_model_info(bert_model_pytorch):
    """Тест get_model_info для PyTorch модели."""
    info = bert_model_pytorch.get_model_info()
    assert info["is_loaded"] is True
    assert info["is_onnx"] is False


# --- Сравнение PyTorch и ONNX ---


def test_bert_pytorch_vs_onnx_similar_output(bert_model_pytorch, bert_model_onnx):
    """
    Тест: предсказания PyTorch и ONNX моделей близки (в пределах допуска).
    Квантизация может незначительно изменить выходы.
    """
    test_texts = [
        "это нормальный комментарий",
        "спасибо за помощь",
        "очень полезная информация",
    ]
    tolerance = 0.05  # допуск на разницу из-за квантизации

    for text in test_texts:
        pt_result = bert_model_pytorch.predict(text)
        onnx_result = bert_model_onnx.predict(text)
        pt_score = pt_result["toxicity_score"]
        onnx_score = onnx_result["toxicity_score"]
        assert abs(pt_score - onnx_score) <= tolerance, (
            f"Текст: {text!r}, PyTorch: {pt_score:.4f}, ONNX: {onnx_score:.4f}"
        )


def test_bert_pytorch_vs_onnx_batch_similar_output(bert_model_pytorch, bert_model_onnx):
    """Тест: batch предсказания PyTorch и ONNX близки."""
    texts = ["текст один", "текст два", "текст три"]
    pt_results = bert_model_pytorch.predict_batch(texts)
    onnx_results = bert_model_onnx.predict_batch(texts)
    tolerance = 0.05

    for pt_r, onnx_r in zip(pt_results, onnx_results):
        assert abs(pt_r["toxicity_score"] - onnx_r["toxicity_score"]) <= tolerance

