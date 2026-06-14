"""Тесты для TextProcessor"""
import pytest
from app.preprocessing.text_processor import TextProcessor


def test_text_processor_normalize():
    """Тест нормализации текста"""
    processor = TextProcessor(use_lemmatization=False, remove_stopwords=False)
    
    text = "Это ТЕКСТ с http://example.com ссылкой"
    normalized = processor.normalize(text)
    
    assert "http" not in normalized
    assert normalized.islower()


def test_text_processor_process():
    """Тест полной обработки текста"""
    processor = TextProcessor()
    
    text = "Это тестовый текст для проверки"
    processed = processor.process(text)
    
    assert isinstance(processed, str)
    assert len(processed) > 0


def test_text_processor_empty_text():
    """Тест обработки пустого текста"""
    processor = TextProcessor()
    
    result = processor.process("")
    assert result == ""
    
    # Проверка обработки пустой строки после нормализации
    result = processor.process("   ")
    assert result == ""


@pytest.mark.parametrize(
    "use_lemmatization,remove_stopwords",
    [
        (False, False),
        (True, True),
        (True, False),
        (False, True),
    ],
)
def test_process_batch_matches_process(use_lemmatization, remove_stopwords):
    """process_batch даёт тот же результат, что и поштучный process."""
    processor = TextProcessor(
        use_lemmatization=use_lemmatization,
        remove_stopwords=remove_stopwords,
    )
    texts = [
        "",
        "   ",
        "Это тестовый текст для проверки",
        "Повтор повтор повтор снова",
        "http://example.com и email@test.ru",
        None,
    ]
    expected = [processor.process(text) for text in texts]
    assert processor.process_batch(texts) == expected

