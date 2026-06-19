"""Предобработка текста для классификации токсичности"""
import re
from functools import lru_cache
from typing import Optional, List

from emoji import replace_emoji
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
import nltk

# Скачиваем стоп-слова при первом импорте
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Инициализация морфологического анализатора
morph = MorphAnalyzer()
emoji_pattern = re.compile(pattern = "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+", 
    flags = re.UNICODE
)

# Загрузка стоп-слов
try:
    stop_words = set(stopwords.words('russian'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('russian'))

# Предкомпилированные шаблоны normalize() — один раз при импорте модуля
# URL: явная схема или www.; не матчим «www» внутри слов; обрезаем на <>"'
_RE_URLS_EMAIL = re.compile(
    r"https?://[^\s<>\"']+"
    r"|www\.[^\s<>\"']+"
    r"|[a-z\d._+-]+@[a-z\d.-]+\.[a-z]{2,}(?:\.[a-z]{2,})?"
)
_RE_HTML_TAGS = re.compile(r"<.*?>")
# VK: [id123|Имя], [club1|Группа] — без .+ (catastrophic backtracking)
_RE_VK_MENTION = re.compile(
    r"\[(?:id|club|public)\d+\|[^\]]+\],?\s*",
    re.IGNORECASE,
)
_RE_HTML_ENTITIES = re.compile(r"&#\d+;|&.+;")
_RE_PUNCTUATION = re.compile(r'[!\"#$%&\'\(\)*+,-./:;<=>?@\[\\\]^_`\{\|\}~]')
_RE_MULTI_SPACE = re.compile(r"\s{2,}")


@lru_cache(maxsize=100000)
def _lemmatize_word(word: str) -> str:
    """Лемматизация одного слова с кэшированием"""
    try:
        return morph.parse(word)[0].normal_form
    except Exception:
        return word

def deEmojify(text):
    return replace_emoji(text, replace=' ')


class TextProcessor:
    """Класс для предобработки текста"""
    
    def __init__(self, use_lemmatization: bool = True, remove_stopwords: bool = True, remove_punkt: bool = True):
        """
        Args:
            use_lemmatization: Использовать ли лемматизацию
            remove_stopwords: Удалять ли стоп-слова
            remove_punkt: Удалять ли пунктуацию
        """
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.remove_punkt = remove_punkt
    
    def process(self, text: str) -> str:
        """
        Основной метод предобработки текста
        
        Args:
            text: Исходный текст
        
        Returns:
            Предобработанный текст
        """
        if text is None or not isinstance(text, str):
            return ""
        
        if not text.strip():
            return ""
        
        # Нормализация
        processed = self.normalize(text)
        
        # Лемматизация (если включена)
        if self.use_lemmatization:
            processed = self.lemmatize(processed)
        
        # Удаление стоп-слов (если включено)
        if self.remove_stopwords:
            processed = self.remove_stop_words(processed)
        
        return processed.strip()
    
    def normalize(self, text: str) -> str:
        """
        Нормализация текста: очистка, приведение к нижнему регистру
        
        Args:
            text: Исходный текст
        
        Returns:
            Нормализованный текст
        """
        # Приведение к нижнему регистру
        text = text.lower()

        text = _RE_URLS_EMAIL.sub(" ", text)
        text = _RE_HTML_TAGS.sub(" ", text)
        text = _RE_VK_MENTION.sub("", text)
        text = _RE_HTML_ENTITIES.sub(" ", text)

        if self.remove_punkt:
            text = _RE_PUNCTUATION.sub(" ", text)

        text = _RE_MULTI_SPACE.sub(" ", text)

        return text.strip()
    
    def lemmatize(self, text: str) -> str:
        """
        Лемматизация текста
        
        Args:
            text: Текст для лемматизации
        
        Returns:
            Лемматизированный текст
        """
        return self._apply_lemma_postprocess(
            ' '.join(_lemmatize_word(w) for w in text.split())
        )

    @staticmethod
    def _apply_lemma_postprocess(text: str) -> str:
        return text.replace('ё', 'е').replace('ъ', 'ь')

    def _lemmatize_texts(self, texts: List[str]) -> List[str]:
        """Лемматизация батча: pymorphy вызывается только для уникальных токенов."""
        unique_words: set[str] = set()
        for text in texts:
            if text:
                unique_words.update(text.split())

        lemma_by_word = {
            word: self._apply_lemma_postprocess(_lemmatize_word(word))
            for word in unique_words
        }

        return [
            ' '.join(lemma_by_word[word] for word in text.split()) if text else ""
            for text in texts
        ]
    
    def remove_stop_words(self, text: str) -> str:
        """
        Удаление стоп-слов
        
        Args:
            text: Текст для обработки
        
        Returns:
            Текст без стоп-слов
        """
        words = text.split()
        filtered = [w for w in words if w not in stop_words]
        return ' '.join(filtered)
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Предобработка батча текстов
        
        Args:
            texts: Список текстов
        
        Returns:
            Список предобработанных текстов
        """
        if not texts:
            return []

        results: List[str] = [""] * len(texts)
        active_indices: List[int] = []
        normalized: List[str] = []

        for index, text in enumerate(texts):
            if text is None or not isinstance(text, str) or not text.strip():
                continue
            active_indices.append(index)
            normalized.append(self.normalize(text))

        if not active_indices:
            return results

        if self.use_lemmatization:
            normalized = self._lemmatize_texts(normalized)

        if self.remove_stopwords:
            normalized = [self.remove_stop_words(text) for text in normalized]

        for index, processed in zip(active_indices, normalized):
            results[index] = processed.strip()

        return results

