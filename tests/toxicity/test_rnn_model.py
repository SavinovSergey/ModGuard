"""Тесты для RNN модели"""
import pytest
import tempfile
from pathlib import Path
import torch
import numpy as np
import json
import shutil

from app.models.rnn_model import RNNModel
from app.models.rnn_tokenizers import BPETokenizer
from app.models.rnn_network import RNNClassifier
from app.preprocessing.text_processor import TextProcessor


@pytest.fixture
def temp_model_files_bpe():
    """Создает временные файлы RNN модели с BPE токенизатором для тестирования"""
    text_processor = TextProcessor(use_lemmatization=False, remove_stopwords=False)
    
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
    processed_texts = [text_processor.normalize(text) for text in texts]
    
    # Обучаем BPE токенизатор
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(processed_texts)
    
    # Создаем простую модель
    vocab_size = tokenizer.get_vocab_size()
    model = RNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_size=16,
        num_layers=1,
        rnn_type='gru',
        dropout=0.1,
        bidirectional=False,
        padding_idx=tokenizer.get_pad_token_id(),
        embedding_dropout=0.0,
        use_layer_norm=False
    )
    
    # Быстрое обучение на небольшом датасете
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Токенизация данных
    token_ids_list = [tokenizer.encode(text, max_length=32) for text in processed_texts]
    input_tensors = torch.tensor(token_ids_list, dtype=torch.long)
    label_tensors = torch.tensor(labels, dtype=torch.float32)
    
    # Несколько шагов обучения
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        logits = model(input_tensors).squeeze(-1)
        loss = criterion(logits, label_tensors)
        loss.backward()
        optimizer.step()
    
    # Сохраняем во временные файлы
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / 'model.pt'
        tokenizer_path = Path(tmpdir) / 'tokenizer.json'
        
        torch.save(model.state_dict(), model_path)
        tokenizer.save(str(tokenizer_path))
        
        # Создаем params.json
        import json
        params_path = Path(tmpdir) / 'params.json'
        params = {
            'tokenizer_type': 'bpe',
            'rnn_type': 'gru',
            'embedding_dim': 32,
            'hidden_size': 16,
            'num_layers': 1,
            'dropout': 0.1,
            'bidirectional': False,
            'max_length': 32,
            'vocab_size': 1000,
            'embedding_dropout': 0.0,
            'use_layer_norm': False
        }
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)
        
        yield str(model_path), str(tokenizer_path)


@pytest.fixture
def rnn_model(temp_model_files_bpe):
    """Создает и загружает RNN модель для тестирования"""
    model_path, tokenizer_path = temp_model_files_bpe
    model = RNNModel()
    model.load(model_path=model_path, tokenizer_path=tokenizer_path)
    return model


@pytest.fixture
def temp_model_files_bpe_quantized(temp_model_files_bpe):
    """Создает квантизованную версию модели из temp_model_files_bpe"""
    model_path, tokenizer_path = temp_model_files_bpe
    model_dir = Path(model_path).parent
    
    # Создаем директорию для квантизированной модели
    quantized_dir = model_dir / "quantized"
    quantized_dir.mkdir(exist_ok=True)
    
    # Копируем токенизатор
    quantized_tokenizer_path = quantized_dir / Path(tokenizer_path).name
    shutil.copy(tokenizer_path, quantized_tokenizer_path)
    
    # Копируем params.json
    params_path = model_dir / "params.json"
    quantized_params_path = quantized_dir / "params.json"
    if params_path.exists():
        shutil.copy(params_path, quantized_params_path)
    
    # Квантизируем модель
    try:
        from scripts.toxicity.quantize_rnn import quantize_rnn_model
    except ImportError:
        pytest.skip("Скрипт квантизации не найден")
    
    quantize_rnn_model(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_dir=quantized_dir,
        device="cpu",
        dtype=torch.qint8
    )
    
    # Возвращаем пути к квантизированной модели
    quantized_model_path = quantized_dir / "model_quantized.pt"
    yield str(quantized_model_path), str(quantized_tokenizer_path)


@pytest.fixture
def rnn_model_pytorch(temp_model_files_bpe):
    """Создает и загружает оригинальную PyTorch RNN модель для тестирования"""
    model_path, tokenizer_path = temp_model_files_bpe
    # Убеждаемся, что params.json не содержит флаг quantized
    model_dir = Path(model_path).parent
    params_path = model_dir / "params.json"
    if params_path.exists():
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        # Удаляем флаг quantized если он есть
        params.pop('quantized', None)
        params.pop('quantization_dtype', None)
        params.pop('quantization_method', None)
        params.pop('embeddings_quantized_fp16', None)
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)
    
    model = RNNModel()
    model.load(model_path=model_path, tokenizer_path=tokenizer_path)
    return model


@pytest.fixture
def rnn_model_quantized(temp_model_files_bpe_quantized):
    """Создает и загружает квантизированную RNN модель для тестирования"""
    model_path, tokenizer_path = temp_model_files_bpe_quantized
    model = RNNModel()
    model.load(model_path=model_path, tokenizer_path=tokenizer_path)
    return model


def test_rnn_model_initialization():
    """Тест инициализации модели"""
    model = RNNModel()
    assert model.model_name == "rnn"
    assert not model.is_loaded
    assert model.model is None
    assert model.tokenizer is None


def test_rnn_model_initialization_with_paths():
    """Тест инициализации модели с путями"""
    model = RNNModel(
        model_path="models/rnn/model.pt",
        tokenizer_path="models/rnn/tokenizer.json"
    )
    assert model.model_name == "rnn"
    assert model.model_path == "models/rnn/model.pt"
    assert model.tokenizer_path == "models/rnn/tokenizer.json"
    assert not model.is_loaded


def test_rnn_model_load(temp_model_files_bpe):
    """Тест загрузки модели"""
    model_path, tokenizer_path = temp_model_files_bpe
    model = RNNModel()
    
    model.load(model_path=model_path, tokenizer_path=tokenizer_path)
    
    assert model.is_loaded
    assert model.model is not None
    assert model.tokenizer is not None
    assert model.model_path == model_path
    assert model.tokenizer_path == tokenizer_path


def test_rnn_model_load_without_paths():
    """Тест загрузки модели без указания путей"""
    model = RNNModel()
    
    with pytest.raises(ValueError, match="Необходимо указать пути"):
        model.load()


def test_rnn_model_load_nonexistent_file():
    """Тест загрузки модели с несуществующим файлом"""
    model = RNNModel()
    
    with pytest.raises(FileNotFoundError):
        model.load(model_path="nonexistent/model.pt", tokenizer_path="nonexistent/tokenizer.json")


def test_rnn_model_predict_not_loaded():
    """Тест предсказания без загрузки модели"""
    model = RNNModel()
    
    with pytest.raises(RuntimeError, match="Модель не загружена"):
        model.predict("тестовый текст")


def test_rnn_model_predict_empty_text(rnn_model):
    """Тест предсказания для пустого текста"""
    result = rnn_model.predict("")
    
    assert result['is_toxic'] is False
    assert result['toxicity_score'] == 0.0
    assert result['toxicity_types'] == {}


def test_rnn_model_predict_whitespace(rnn_model):
    """Тест предсказания для текста только с пробелами"""
    result = rnn_model.predict("   ")
    
    assert result['is_toxic'] is False
    assert result['toxicity_score'] == 0.0


def test_rnn_model_predict_normal_text(rnn_model):
    """Тест предсказания для нормального текста"""
    result = rnn_model.predict("это нормальный комментарий")
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result
    assert 'toxicity_types' in result
    assert isinstance(result['toxicity_score'], float)
    assert 0.0 <= result['toxicity_score'] <= 1.0


def test_rnn_model_predict_toxic_text(rnn_model):
    """Тест предсказания для токсичного текста"""
    result = rnn_model.predict("иди нахуй")
    
    assert isinstance(result, dict)
    assert 'is_toxic' in result
    assert 'toxicity_score' in result
    assert isinstance(result['toxicity_score'], float)
    assert 0.0 <= result['toxicity_score'] <= 1.0


def test_rnn_model_predict_batch_not_loaded():
    """Тест batch предсказания без загрузки модели"""
    model = RNNModel()
    
    with pytest.raises(RuntimeError, match="Модель не загружена"):
        model.predict_batch(["текст 1", "текст 2"])


def test_rnn_model_predict_batch_empty_list(rnn_model):
    """Тест batch предсказания для пустого списка"""
    results = rnn_model.predict_batch([])
    
    assert results == []


def test_rnn_model_predict_batch(rnn_model):
    """Тест batch предсказания"""
    texts = [
        "это нормальный комментарий",
        "иди нахуй",
        "спасибо за информацию"
    ]
    
    results = rnn_model.predict_batch(texts)
    
    assert len(results) == 3
    for result in results:
        assert isinstance(result, dict)
        assert 'is_toxic' in result
        assert 'toxicity_score' in result
        assert 'toxicity_types' in result
        assert isinstance(result['toxicity_score'], float)
        assert 0.0 <= result['toxicity_score'] <= 1.0


def test_rnn_model_predict_batch_with_empty_texts(rnn_model):
    """Тест batch предсказания с пустыми текстами"""
    texts = ["", "   ", "нормальный текст"]
    
    results = rnn_model.predict_batch(texts)
    
    assert len(results) == 3
    # Пустые тексты должны возвращать is_toxic=False
    assert results[0]['is_toxic'] is False
    assert results[1]['is_toxic'] is False


def test_rnn_model_info_not_loaded():
    """Тест получения информации о незагруженной модели"""
    model = RNNModel()
    info = model.get_model_info()
    
    assert info['name'] == 'rnn'
    assert info['type'] == 'rnn'
    assert info['is_loaded'] is False
    assert 'version' in info
    assert 'description' in info


def test_rnn_model_info_loaded(rnn_model):
    """Тест получения информации о загруженной модели"""
    info = rnn_model.get_model_info()
    
    assert info['name'] == 'rnn'
    assert info['type'] == 'rnn'
    assert info['is_loaded'] is True
    assert 'version' in info
    assert 'description' in info
    assert 'tokenizer_type' in info
    assert 'rnn_type' in info
    assert 'model_params' in info


def test_rnn_model_predict_consistency(rnn_model):
    """Тест консистентности предсказаний"""
    text = "тестовый текст для проверки"
    
    result1 = rnn_model.predict(text)
    result2 = rnn_model.predict(text)
    
    # Предсказания должны быть одинаковыми
    assert result1['is_toxic'] == result2['is_toxic']
    assert abs(result1['toxicity_score'] - result2['toxicity_score']) < 1e-6


def test_rnn_model_load_saves_paths(temp_model_files_bpe):
    """Тест что пути сохраняются после загрузки"""
    model_path, tokenizer_path = temp_model_files_bpe
    model = RNNModel()
    
    model.load(model_path=model_path, tokenizer_path=tokenizer_path)
    
    # Пути должны быть сохранены
    assert model.model_path == model_path
    assert model.tokenizer_path == tokenizer_path


def test_rnn_model_load_with_instance_paths(temp_model_files_bpe):
    """Тест загрузки модели с путями, указанными при инициализации"""
    model_path, tokenizer_path = temp_model_files_bpe
    model = RNNModel(model_path=model_path, tokenizer_path=tokenizer_path)
    
    # Загрузка без параметров должна использовать сохраненные пути
    model.load()
    
    assert model.is_loaded
    assert model.model is not None
    assert model.tokenizer is not None


# --- Тесты для квантизированной RNN модели ---


def test_rnn_model_quantized_predict(rnn_model_quantized):
    """Тест предсказания квантизированной RNN модели."""
    result = rnn_model_quantized.predict("это тестовый комментарий")
    assert isinstance(result, dict)
    assert "is_toxic" in result
    assert "toxicity_score" in result
    assert "toxicity_types" in result
    assert isinstance(result["toxicity_score"], float)
    assert 0.0 <= result["toxicity_score"] <= 1.0


def test_rnn_model_quantized_predict_batch(rnn_model_quantized):
    """Тест batch предсказания квантизированной RNN модели."""
    texts = ["нормальный текст", "токсичный комментарий"]
    results = rnn_model_quantized.predict_batch(texts)
    assert len(results) == 2
    for r in results:
        assert "toxicity_score" in r
        assert 0.0 <= r["toxicity_score"] <= 1.0


def test_rnn_model_quantized_get_model_info(rnn_model_quantized):
    """Тест get_model_info для квантизированной RNN модели."""
    info = rnn_model_quantized.get_model_info()
    assert info["is_loaded"] is True
    assert "is_quantized" in info
    assert info["is_quantized"] is True
    assert "quantization_dtype" in info
    assert "quantization_method" in info
    assert "model_params" in info


def test_rnn_model_quantized_info_has_quantization_details(rnn_model_quantized):
    """Тест что информация о модели содержит детали квантизации."""
    info = rnn_model_quantized.get_model_info()
    assert "is_quantized" in info
    if info["is_quantized"]:
        assert "quantization_dtype" in info
        assert "quantization_method" in info
        # Проверяем, что embeddings_quantized_fp16 присутствует, если была применена квантизация эмбеддингов
        model_params = info.get("model_params", {})
        if "embeddings_quantized_fp16" in model_params:
            assert isinstance(model_params["embeddings_quantized_fp16"], bool)


# --- Тесты для оригинальной PyTorch RNN модели ---


def test_rnn_model_pytorch_get_model_info(rnn_model_pytorch):
    """Тест get_model_info для оригинальной PyTorch RNN модели."""
    info = rnn_model_pytorch.get_model_info()
    assert info["is_loaded"] is True
    # Оригинальная модель не должна быть квантизирована
    if "is_quantized" in info:
        assert info["is_quantized"] is False


# --- Сравнение оригинальной и квантизированной RNN моделей ---


def test_rnn_pytorch_vs_quantized_similar_output(rnn_model_pytorch, rnn_model_quantized):
    """
    Тест: предсказания оригинальной и квантизированной RNN моделей близки (в пределах допуска).
    Квантизация может незначительно изменить выходы.
    """
    test_texts = [
        "это нормальный комментарий",
        "спасибо за помощь",
        "очень полезная информация",
    ]
    tolerance = 0.1  # допуск на разницу из-за квантизации (больше чем для ONNX, т.к. динамическая квантизация менее точна)

    for text in test_texts:
        pt_result = rnn_model_pytorch.predict(text)
        quantized_result = rnn_model_quantized.predict(text)
        pt_score = pt_result["toxicity_score"]
        quantized_score = quantized_result["toxicity_score"]
        assert abs(pt_score - quantized_score) <= tolerance, (
            f"Текст: {text!r}, PyTorch: {pt_score:.4f}, Квантизированная: {quantized_score:.4f}"
        )


def test_rnn_pytorch_vs_quantized_batch_similar_output(rnn_model_pytorch, rnn_model_quantized):
    """Тест: batch предсказания оригинальной и квантизированной RNN моделей близки."""
    texts = ["текст один", "текст два", "текст три"]
    pt_results = rnn_model_pytorch.predict_batch(texts)
    quantized_results = rnn_model_quantized.predict_batch(texts)
    tolerance = 0.1

    for pt_r, quantized_r in zip(pt_results, quantized_results):
        assert abs(pt_r["toxicity_score"] - quantized_r["toxicity_score"]) <= tolerance, (
            f"PyTorch: {pt_r['toxicity_score']:.4f}, Квантизированная: {quantized_r['toxicity_score']:.4f}"
        )


def test_rnn_model_quantized_loads_correctly(temp_model_files_bpe_quantized):
    """Тест что квантизированная модель загружается корректно."""
    model_path, tokenizer_path = temp_model_files_bpe_quantized
    model = RNNModel()
    model.load(model_path=model_path, tokenizer_path=tokenizer_path)
    
    assert model.is_loaded
    assert model.model is not None
    assert model.tokenizer is not None
    
    # Проверяем, что модель действительно квантизирована
    info = model.get_model_info()
    assert info.get("is_quantized", False) is True



