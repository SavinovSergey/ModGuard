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


def test_normalize_strips_urls_and_emails():
    processor = TextProcessor(use_lemmatization=False, remove_stopwords=False)

    assert "http" not in processor.normalize("Ссылка https://example.com/path ок")
    assert "www" not in processor.normalize("смотри www.site.ru/page")
    assert "mail.ru" not in processor.normalize("пиши user@mail.ru сюда")
    # не вырезаем «www» внутри слова
    assert "awwwesome" in processor.normalize("фильм awwwesome")
    # URL не съедает <br> и текст после (граница на <>)
    out = processor.normalize("перейди https://t.me/x<br>иди дальше")
    assert "иди" in out
    assert "дальше" in out


def test_normalize_strips_vk_mentions():
    processor = TextProcessor(use_lemmatization=False, remove_stopwords=False)

    out = processor.normalize("[id709526502|анна], напишите в личку")
    assert "[id" not in out
    assert "анна" not in out
    assert "напишите" in out

    out = processor.normalize("привет [id747854376|дарья фомина]")
    assert "привет" in out
    assert "дарья" not in out

    out = processor.normalize("[id695319329|элена санг],вам хотелось")
    assert "вам" in out
    assert "элена" not in out


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

