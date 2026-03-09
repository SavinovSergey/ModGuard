"""Тесты для спам TF-IDF модели (SpamTfidfModel)."""
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from app.models.spam.tfidf_model import SpamTfidfModel
from app.preprocessing.spam_processor import SpamTextProcessor
from .spam_quality_examples import SPAM_QUALITY_HAM_75, SPAM_QUALITY_SPAM_25

# 20 нормальных (ham) и 10 спам-сообщений для быстрого теста качества (F1 >= 0.95)
QUALITY_TEST_HAM = [
    "Спасибо за информацию, очень полезно.",
    "Согласен с предыдущим оратором.",
    "Интересная точка зрения, не думал об этом.",
    "Какой у вас опыт с этим продуктом?",
    "Подскажите, где можно посмотреть инструкцию?",
    "Хорошая статья, сохранил в закладки.",
    "Давно искал такой ответ, благодарю.",
    "Не совсем понял последний абзац, можно подробнее?",
    "Полностью поддерживаю.",
    "У меня похожая ситуация, как решили?",
    "Напишите, если будут ещё материалы по теме.",
    "Это работает в последней версии?",
    "Ок, принял к сведению.",
    "Коллеги, кто уже пробовал этот способ?",
    "Нашёл решение в документации.",
    "Спасибо, что поделились опытом.",
    "Жду продолжения обсуждения.",
    "Всё верно, так и есть.",
    "Полезный комментарий, учту.",
    "Ничего не сломалось после обновления?",
]

QUALITY_TEST_SPAM = [
    "БЕСПЛАТНО!!! Переходи по ссылке https://bit.ly/xyz и забери подарок",
    "Заработок без вложений за один день!!! Ссылка в профиле",
    "Смотри бесплатно тут: http://spam-site.xyz/page?ref=123",
    "КАЖДОМУ ПОДАРОК!!! Регистрируйся по ссылке goo.gl/spam",
    "Бесплатные подарки только сегодня!!! https://promo-site.top",
    "Заработай 100000 за час, детали в личку!!! bit.ly/money",
    "Смотри что скрывают!!! Перейди по ссылке tinyurl.com/secret",
    "Акция!!! Бесплатно получи товар, переходи по ссылке в описании",
    "Не упусти!!! Бесплатная раздача, ссылка в шапке профиля",
]


@pytest.fixture
def spam_temp_model_dir():
    """Временная директория с model.pkl и vectorizer.pkl (режим только TF-IDF, без фичей)."""
    processor = SpamTextProcessor()
    texts = [
        "обычное сообщение без спама",
        "бесплатно перейди по ссылке https://spam.com",
        "спасибо за ответ",
        "КАПС И ЕЩЁ ССЫЛКА bit.ly/xyz",
        "нормальный комментарий",
    ]
    labels = [0, 1, 0, 1, 0]  # 0 = ham, 1 = spam
    processed = [processor.process(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, labels)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        with open(p / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(p / "vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        yield str(p)


@pytest.fixture
def spam_model(spam_temp_model_dir):
    """Загруженная спам-модель (только TF-IDF)."""
    m = SpamTfidfModel()
    m.load(
        model_path=f"{spam_temp_model_dir}/model.pkl",
        vectorizer_path=f"{spam_temp_model_dir}/vectorizer.pkl",
    )
    return m


def test_spam_model_init():
    """Инициализация: не загружена, порог 0.5."""
    model = SpamTfidfModel()
    assert model.is_loaded is False
    assert model.model is None
    assert model.vectorizer is None
    assert model.optimal_threshold == 0.5
    assert model.use_extra_features is False
    assert model.scaler is None


def test_spam_model_load(spam_temp_model_dir):
    """Загрузка модели и векторизатора."""
    model = SpamTfidfModel()
    model.load(
        model_path=f"{spam_temp_model_dir}/model.pkl",
        vectorizer_path=f"{spam_temp_model_dir}/vectorizer.pkl",
    )
    assert model.is_loaded is True
    assert model.model is not None
    assert model.vectorizer is not None
    assert isinstance(model.model, LogisticRegression)
    assert isinstance(model.vectorizer, TfidfVectorizer)


def test_spam_model_load_nonexistent_model():
    """Загрузка при отсутствующем model.pkl."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        with open(p / "vectorizer.pkl", "wb") as f:
            pickle.dump(TfidfVectorizer(max_features=10), f)
        m = SpamTfidfModel()
        with pytest.raises(FileNotFoundError, match="модели не найден"):
            m.load(
                model_path=str(p / "model.pkl"),
                vectorizer_path=str(p / "vectorizer.pkl"),
            )


def test_spam_model_load_nonexistent_vectorizer(spam_temp_model_dir):
    """Загрузка при отсутствующем vectorizer.pkl."""
    m = SpamTfidfModel()
    with pytest.raises(FileNotFoundError, match="векторизатора не найден"):
        m.load(
            model_path=f"{spam_temp_model_dir}/model.pkl",
            vectorizer_path=f"{spam_temp_model_dir}/nonexistent.pkl",
        )


def test_spam_model_predict_not_loaded():
    """Предсказание без загрузки возвращает пустой ответ."""
    model = SpamTfidfModel()
    out = model.predict("любой текст")
    assert out == {"is_spam": False, "spam_score": 0.0}


def test_spam_model_predict_empty_text(spam_model):
    """Пустой текст -> is_spam=False, spam_score=0."""
    out = spam_model.predict("")
    assert out["is_spam"] is False
    assert out["spam_score"] == 0.0


def test_spam_model_predict_whitespace(spam_model):
    """Только пробелы -> пустой ответ."""
    out = spam_model.predict("   \n\t  ")
    assert out["is_spam"] is False
    assert out["spam_score"] == 0.0


def test_spam_model_predict_returns_structure(spam_model):
    """predict возвращает dict с is_spam и spam_score."""
    out = spam_model.predict("обычное сообщение без спама")
    assert isinstance(out, dict)
    assert "is_spam" in out
    assert "spam_score" in out
    assert isinstance(out["is_spam"], bool)
    assert isinstance(out["spam_score"], float)
    assert 0.0 <= out["spam_score"] <= 1.0


def test_spam_model_predict_batch_not_loaded():
    """predict_batch без загрузки возвращает пустые ответы по числу текстов."""
    model = SpamTfidfModel()
    out = model.predict_batch(["a", "b"])
    assert len(out) == 2
    assert all(r["is_spam"] is False and r["spam_score"] == 0.0 for r in out)


def test_spam_model_predict_batch_empty_list(spam_model):
    """predict_batch([]) -> []."""
    out = spam_model.predict_batch([])
    assert out == []


def test_spam_model_predict_batch(spam_model):
    """predict_batch возвращает один результат на текст."""
    texts = ["обычное сообщение", "ещё один текст", "третий"]
    out = spam_model.predict_batch(texts)
    assert len(out) == 3
    for r in out:
        assert "is_spam" in r and "spam_score" in r
        assert isinstance(r["spam_score"], float)
        assert 0.0 <= r["spam_score"] <= 1.0


def test_spam_model_predict_batch_with_empty_strings(spam_model):
    """В батче пустые строки дают is_spam=False, spam_score=0."""
    texts = ["", "   ", "нормальный текст"]
    out = spam_model.predict_batch(texts)
    assert len(out) == 3
    assert out[0]["is_spam"] is False and out[0]["spam_score"] == 0.0
    assert out[1]["is_spam"] is False and out[1]["spam_score"] == 0.0
    assert "spam_score" in out[2] and 0.0 <= out[2]["spam_score"] <= 1.0


def test_spam_model_predict_consistency(spam_model):
    """Один и тот же текст даёт одинаковый результат."""
    text = "тестовый текст для консистентности"
    a = spam_model.predict(text)
    b = spam_model.predict(text)
    assert a["is_spam"] == b["is_spam"]
    assert abs(a["spam_score"] - b["spam_score"]) < 1e-9


def test_spam_model_load_with_params(spam_temp_model_dir):
    """Загрузка с params.json: optimal_threshold подхватывается."""
    p = Path(spam_temp_model_dir)
    with open(p / "params.json", "w", encoding="utf-8") as f:
        json.dump({"optimal_threshold": 0.3}, f)
    m = SpamTfidfModel()
    m.load(model_path=str(p / "model.pkl"), vectorizer_path=str(p / "vectorizer.pkl"))
    assert m.optimal_threshold == 0.3


def test_spam_model_use_extra_features_false_without_params(spam_model):
    """Без params.json use_extra_features остаётся False."""
    assert spam_model.use_extra_features is False
    assert spam_model.scaler is None





def test_spam_model_quality_f1_extended():
    """Качество на 75 ham + 25 spam: F1 >= 0.95. Расширенный набор из tests/spam_quality_examples.py."""
    model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "spam" / "tfidf"
    model_path = model_dir / "model.pkl"
    vectorizer_path = model_dir / "vectorizer.pkl"
    if not model_path.exists() or not vectorizer_path.exists():
        pytest.skip("Обученная модель models/spam/tfidf не найдена (запустите обучение спам-модели)")

    model = SpamTfidfModel()
    model.load(model_path=str(model_path), vectorizer_path=str(vectorizer_path))
    assert model.is_loaded and model.optimal_threshold is not None

    texts = SPAM_QUALITY_HAM_75 + SPAM_QUALITY_SPAM_25
    y_true = [0] * len(SPAM_QUALITY_HAM_75) + [1] * len(SPAM_QUALITY_SPAM_25)

    results = model.predict_batch(texts)
    y_pred = [1 if r["is_spam"] else 0 for r in results]

    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    assert f1 >= 0.9, (
        f"F1 на расширенном наборе (75 ham, 25 spam) = {f1:.4f}, ожидается >= 0.95. "
        f"Precision = {precision:.4f}, Recall = {recall:.4f}. "
        f"Порог = {model.optimal_threshold}. "
        f"Ошибки: {[(pred, true, t[:50]) for pred, true, t in zip(y_pred, y_true, texts) if pred != true]}"
    )
