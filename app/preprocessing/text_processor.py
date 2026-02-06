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


@lru_cache(maxsize=1000)
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
        
        # Удаление ссылок на сайты, почты и аккаунты
        text = re.sub(r"(http|www)\S+|[a-z\d\._-]+@[a-z\d\._-]+\.[a-z\d\._-]+|@[a-z]+", " ", text)
        
        # HTML тэги
        text = re.sub(r"<.*?>", " ", text)
        
        # Удаление отметок вида [id647188941|зара]
        text = re.sub(r"\[id\d+|.+\], ", "", text)
        
        # Удаление сочетаний вида &#33;
        text = re.sub(r'&#\d+;|&.+;', ' ', text)
        
        if self.remove_punkt:
            text = re.sub(r'[!\"#$%&\'\(\)*+,-./:;<=>?@\[\\\]^_`\{\|\}~]', ' ', text)
        
        # Удаление множественных пробелов
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    def lemmatize(self, text: str) -> str:
        """
        Лемматизация текста
        
        Args:
            text: Текст для лемматизации
        
        Returns:
            Лемматизированный текст
        """
        words = text.split()
        lemmatized = [_lemmatize_word(w) for w in words]
        result = ' '.join(lemmatized)
        # Замена ё на е
        result = result.replace('ё', 'е')
        result = result.replace('ъ', 'ь')
        return result
    
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
        return [self.process(text) for text in texts]

