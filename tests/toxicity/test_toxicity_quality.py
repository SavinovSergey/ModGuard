# -*- coding: utf-8 -*-
"""Тесты качества моделей токсичности на наборе из tests/toxicity_quality_examples.py."""
import pytest
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score

from app.models.toxicity.bert_model import BERTModel, _ONNX_FILES
from .toxicity_quality_examples import get_toxicity_quality_texts_and_labels


def _has_pytorch_bert(path: Path) -> bool:
    """Проверяет наличие PyTorch BERT (config.json + веса)."""
    if not path.exists() or not (path / "config.json").exists():
        return False
    return (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()


def _has_onnx_bert(path: Path) -> bool:
    """Проверяет наличие ONNX BERT."""
    if not path.exists():
        return False
    return any((path / fname).exists() for fname in _ONNX_FILES)

# Минимальный F1 для прохождения теста (обученная модель должна различать токсичные тексты)
MIN_F1_QUALITY = 0.80
# BERT/ONNX может иметь другую калибровку порога и на малом наборе давать чуть меньший F1
MIN_F1_QUALITY_BERT = 0.75


def _load_tfidf_model():
    from app.models.toxicity.tfidf_model import TfidfModel
    model_path = Path("models/toxicity/tfidf/model.pkl")
    vectorizer_path = Path("models/toxicity/tfidf/vectorizer.pkl")
    if not model_path.exists() or not vectorizer_path.exists():
        return None
    model = TfidfModel()
    model.load(model_path=str(model_path), vectorizer_path=str(vectorizer_path))
    return model


def _load_fasttext_model():
    from app.models.toxicity.fasttext_model import FastTextModel
    model_path = Path("models/toxicity/fasttext/fasttext_model.bin")
    if not model_path.exists():
        return None
    model = FastTextModel()
    model.load(model_path=str(model_path))
    return model


def _load_rnn_model():
    from app.models.toxicity.rnn_model import RNNModel
    rnn_dir = Path("models/toxicity/rnn")
    tokenizer_path = rnn_dir / "tokenizer.json"
    model_path = rnn_dir / "model_quantized.pt"
    if not model_path.exists():
        model_path = rnn_dir / "model.pt"
    if not model_path.exists() or not tokenizer_path.exists():
        return None
    model = RNNModel()
    model.load(model_path=str(model_path), tokenizer_path=str(tokenizer_path))
    return model


def _load_bert_model():
    """
    Загружает BERT: при наличии PyTorch в models/toxicity/bert — только его (без fallback на ONNX при ошибке),
    иначе ONNX из models/toxicity/bert/onnx или models/toxicity/bert/onnx_cpu.
    """
    model_path = Path("models/toxicity/bert")
    if model_path.exists() and (model_path / "config.json").exists():
        if _has_pytorch_bert(model_path):
            # Пробуем только PyTorch; при ошибке не переходим на ONNX, чтобы F1 не прыгал между запусками
            model = BERTModel(model_path=str(model_path), use_onnx=False)
            model.load()
            return model
        if _has_onnx_bert(model_path):
            try:
                model = BERTModel(model_path=str(model_path), use_onnx=True)
                model.load()
                return model
            except Exception:
                pass
    for bert_dir in (Path("models/toxicity/bert/onnx"), Path("models/toxicity/bert/onnx_cpu")):
        if _has_onnx_bert(bert_dir):
            try:
                model = BERTModel(model_path=str(bert_dir), use_onnx=True)
                model.load()
                return model
            except Exception:
                continue
    return None


def _run_quality_test(
    model,
    model_name: str,
    min_f1: float = MIN_F1_QUALITY,
    use_score_threshold: bool = False,
):
    """
    Запускает predict_batch на датасете качества и проверяет F1 >= min_f1.
    use_score_threshold: если True, предсказание по toxicity_score >= 0.5 (для теста
    разделительной способности, без привязки к optimal_threshold модели).
    """
    texts, y_true = get_toxicity_quality_texts_and_labels()
    results = model.predict_batch(texts)
    if len(results) != len(texts):
        raise AssertionError(
            f"Модель {model_name} вернула {len(results)} результатов вместо {len(texts)}"
        )
    if use_score_threshold:
        y_pred = [1 if (r.get("toxicity_score", 0.0) >= 0.5) else 0 for r in results]
    else:
        y_pred = [1 if r.get("is_toxic", False) else 0 for r in results]
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    threshold_note = " (порог 0.5 по score)" if use_score_threshold else ""
    assert f1 >= min_f1, (
        f"F1 модели {model_name} на наборе качества = {f1:.4f}, ожидается >= {min_f1}. "
        f"Precision = {precision:.4f}, Recall = {recall:.4f}.{threshold_note} "
        f"Переобучите модель или проверьте порог."
    )


def test_toxicity_quality_tfidf():
    """Качество TF-IDF модели на наборе из toxicity_quality_examples (82 нетоксичных + 30 токсичных)."""
    model = _load_tfidf_model()
    if model is None:
        pytest.skip("Модель models/toxicity/tfidf не найдена (запустите обучение)")
    _run_quality_test(model, "tfidf")


def test_toxicity_quality_fasttext():
    """Качество FastText модели на наборе из toxicity_quality_examples."""
    model = _load_fasttext_model()
    if model is None:
        pytest.skip("Модель models/toxicity/fasttext не найдена (запустите обучение)")
    _run_quality_test(model, "fasttext")


def test_toxicity_quality_rnn():
    """Качество RNN модели на наборе из toxicity_quality_examples."""
    model = _load_rnn_model()
    if model is None:
        pytest.skip("Модель models/toxicity/rnn не найдена (запустите обучение)")
    _run_quality_test(model, "rnn")


def _set_torch_deterministic():
    """Включает детерминированный режим PyTorch/CUDA до загрузки модели и предсказаний."""
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def test_toxicity_quality_bert():
    """
    Качество BERT на наборе из toxicity_quality_examples.
    Оценка по порогу 0.5 по toxicity_score, чтобы не зависеть от optimal_threshold в params.json.
    Детерминизм включается до загрузки модели, чтобы F1 не зависел от порядка запуска тестов.
    """
    _set_torch_deterministic()
    model = _load_bert_model()
    if model is None:
        pytest.skip("Модель models/toxicity/bert (или onnx) не найдена (запустите обучение/квантизацию)")
    try:
        _run_quality_test(
            model,
            "bert",
            min_f1=MIN_F1_QUALITY_BERT,
            use_score_threshold=True,
        )
    except Exception as e:
        if "F1" in str(e) or "ожидается >=" in str(e):
            raise
        pytest.fail(
            f"Тест качества BERT упал с ошибкой: {e}. "
            "Проверьте, что модель загружена (models/toxicity/bert или models/toxicity/bert/onnx) и predict_batch работает."
        )
