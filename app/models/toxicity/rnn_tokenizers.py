"""Токенизаторы для RNN модели"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union
import json

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


class BaseRNNTokenizer(ABC):
    """Базовый класс для токенизаторов RNN"""
    
    @abstractmethod
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Кодирует текст в последовательность токенов
        
        Args:
            text: Текст для кодирования
            max_length: Максимальная длина последовательности (обрезание/паттинг)
        
        Returns:
            Список индексов токенов
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Декодирует последовательность токенов в текст
        
        Args:
            token_ids: Список индексов токенов
        
        Returns:
            Декодированный текст
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """
        Возвращает размер словаря
        
        Returns:
            Размер словаря
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Сохраняет токенизатор
        
        Args:
            path: Путь для сохранения
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Загружает токенизатор
        
        Args:
            path: Путь к файлу токенизатора
        """
        pass

    @abstractmethod
    def get_pad_token_id(self) -> int:
        """
        Возвращает id pad-токена.
        """
        pass

    @abstractmethod
    def get_unk_token_id(self) -> int:
        """
        Возвращает id unk-токена.
        """
        pass


class BPETokenizer(BaseRNNTokenizer):
    """BPE токенизатор на основе библиотеки tokenizers"""
    
    def __init__(self, vocab_size: int = 20000):
        """
        Args:
            vocab_size: Размер словаря
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "Библиотека tokenizers не установлена. "
                "Установите её командой: pip install tokenizers"
            )
        
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._is_trained = False
    
    def train(self, texts: List[str]) -> None:
        """
        Обучает BPE токенизатор на корпусе текстов
        
        Args:
            texts: Список текстов для обучения
        """
        if not texts:
            raise ValueError("Список текстов для обучения не может быть пустым")
        
        # Создаем токенизатор
        self.tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        # Комбинируем нормализаторы используя Sequence вместо оператора |
        self.tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Обучаем токенизатор
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
        )
        
        # Подготовка данных для обучения
        def get_training_corpus():
            for text in texts:
                yield text
        
        self.tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        
        # Добавляем пост-процессор для добавления специальных токенов
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", self.tokenizer.token_to_id("<BOS>")),
                ("<EOS>", self.tokenizer.token_to_id("<EOS>")),
            ],
        )
        
        self._is_trained = True
        logger.info(f"BPE токенизатор обучен, размер словаря: {self.get_vocab_size()}")
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Кодирует текст в последовательность токенов"""
        if not self._is_trained or self.tokenizer is None:
            raise RuntimeError("Токенизатор не обучен. Вызовите train() сначала.")
        
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        
        if max_length is not None:
            # Обрезаем или добавляем padding.
            # При обрезании сохраняем BOS/EOS для корректной структуры последовательности.
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                bos_id = self.tokenizer.token_to_id("<BOS>")
                eos_id = self.tokenizer.token_to_id("<EOS>")
                if max_length >= 1 and bos_id is not None:
                    token_ids[0] = bos_id
                if max_length >= 2 and eos_id is not None:
                    token_ids[-1] = eos_id
            else:
                pad_id = self.tokenizer.token_to_id("<PAD>")
                if pad_id is None:
                    pad_id = 0  # Fallback
                token_ids = token_ids + [pad_id] * (max_length - len(token_ids))
        
        return token_ids
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        """
        Кодирует батч текстов
        
        Args:
            texts: Список текстов
            max_length: Максимальная длина последовательности
        
        Returns:
            Список списков токенов
        """
        return [self.encode(text, max_length) for text in texts]
    
    def decode(self, token_ids: List[int]) -> str:
        """Декодирует последовательность токенов в текст"""
        if not self._is_trained or self.tokenizer is None:
            raise RuntimeError("Токенизатор не обучен. Вызовите train() сначала.")
        
        # Удаляем специальные токены перед декодированием
        pad_id = self.tokenizer.token_to_id("<PAD>")
        bos_id = self.tokenizer.token_to_id("<BOS>")
        eos_id = self.tokenizer.token_to_id("<EOS>")
        
        filtered_ids = [
            tid for tid in token_ids 
            if tid not in [pad_id, bos_id, eos_id] and tid is not None
        ]
        
        return self.tokenizer.decode(filtered_ids)
    
    def get_vocab_size(self) -> int:
        """Возвращает размер словаря"""
        if not self._is_trained or self.tokenizer is None:
            return self.vocab_size
        return self.tokenizer.get_vocab_size()
    
    def save(self, path: str) -> None:
        """Сохраняет токенизатор"""
        if not self._is_trained or self.tokenizer is None:
            raise RuntimeError("Токенизатор не обучен. Невозможно сохранить.")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем токенизатор
        self.tokenizer.save(str(path_obj))
        
        # Сохраняем метаданные
        metadata = {
            'vocab_size': self.vocab_size,
            'type': 'bpe'
        }
        metadata_path = path_obj.parent / f"{path_obj.stem}_config.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"BPE токенизатор сохранен: {path}")
    
    def load(self, path: str) -> None:
        """Загружает токенизатор"""
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "Библиотека tokenizers не установлена. "
                "Установите её командой: pip install tokenizers"
            )
        
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Файл токенизатора не найден: {path}")
        
        # Загружаем токенизатор
        self.tokenizer = Tokenizer.from_file(str(path_obj))
        
        # Загружаем метаданные если есть
        metadata_path = path_obj.parent / f"{path_obj.stem}_config.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.vocab_size = metadata.get('vocab_size', self.vocab_size)
        
        self._is_trained = True
        logger.info(f"BPE токенизатор загружен из {path}")

    def get_pad_token_id(self) -> int:
        """Возвращает id pad-токена."""
        if not self._is_trained or self.tokenizer is None:
            return 0
        pad_id = self.tokenizer.token_to_id("<PAD>")
        return 0 if pad_id is None else int(pad_id)

    def get_unk_token_id(self) -> int:
        """Возвращает id unk-токена."""
        if not self._is_trained or self.tokenizer is None:
            return 0
        unk_id = self.tokenizer.token_to_id("<UNK>")
        return 0 if unk_id is None else int(unk_id)


class RuBERTTokenizer(BaseRNNTokenizer):
    """Обертка над токенизатором ruBERT из transformers"""
    
    def __init__(self, model_name: str = "DeepPavlov/rubert-base-cased"):
        """
        Args:
            model_name: Имя модели ruBERT из HuggingFace
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Библиотека transformers не установлена. "
                "Установите её командой: pip install transformers"
            )
        
        self.model_name = model_name
        self.tokenizer = None
        self._is_loaded = False
    
    def _ensure_loaded(self):
        """Убеждается, что токенизатор загружен"""
        if not self._is_loaded or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._is_loaded = True
            logger.info(f"RuBERT токенизатор загружен: {self.model_name}")
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Кодирует текст в последовательность токенов"""
        self._ensure_loaded()
        
        # Используем encode с параметрами
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length' if max_length else False,
            truncation=max_length is not None,
            return_tensors=None  # Возвращаем список, не тензор
        )
        
        return encoding['input_ids']
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        """
        Кодирует батч текстов
        
        Args:
            texts: Список текстов
            max_length: Максимальная длина последовательности
        
        Returns:
            Список списков токенов
        """
        self._ensure_loaded()
        
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length' if max_length else True,
            truncation=max_length is not None,
            return_tensors=None
        )
        
        return encodings['input_ids']
    
    def decode(self, token_ids: List[int]) -> str:
        """Декодирует последовательность токенов в текст"""
        self._ensure_loaded()
        
        # Удаляем специальные токены (PAD, CLS, SEP) перед декодированием
        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        
        filtered_ids = [
            tid for tid in token_ids 
            if tid not in [pad_id, cls_id, sep_id] and tid is not None
        ]
        
        return self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Возвращает размер словаря"""
        self._ensure_loaded()
        return len(self.tokenizer)
    
    def save(self, path: str) -> None:
        """Сохраняет конфигурацию токенизатора (имя модели)"""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'model_name': self.model_name,
            'type': 'rubert'
        }
        
        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"RuBERT токенизатор конфигурация сохранена: {path}")
    
    def load(self, path: str) -> None:
        """Загружает конфигурацию токенизатора"""
        path_obj = Path(path)
        if path_obj.exists():
            with open(path_obj, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.model_name = metadata.get('model_name', self.model_name)
            # Токенизатор загрузится при первом использовании
            self._is_loaded = False
            logger.info(f"RuBERT токенизатор конфигурация загружена: {path}")
        else:
            # Если файл не существует, используем имя модели по умолчанию
            logger.warning(f"Файл конфигурации токенизатора не найден: {path}, используем модель по умолчанию")
            self._is_loaded = False

    def get_pad_token_id(self) -> int:
        """Возвращает id pad-токена."""
        self._ensure_loaded()
        if self.tokenizer.pad_token_id is None:
            return 0
        return int(self.tokenizer.pad_token_id)

    def get_unk_token_id(self) -> int:
        """Возвращает id unk-токена."""
        self._ensure_loaded()
        if self.tokenizer.unk_token_id is None:
            return 0
        return int(self.tokenizer.unk_token_id)


def create_tokenizer(tokenizer_type: str, **kwargs) -> BaseRNNTokenizer:
    """
    Фабричная функция для создания токенизатора
    
    Args:
        tokenizer_type: Тип токенизатора ('bpe' или 'rubert')
        **kwargs: Дополнительные параметры для токенизатора
    
    Returns:
        Экземпляр токенизатора
    """
    if tokenizer_type.lower() == 'bpe':
        vocab_size = kwargs.get('vocab_size', 20000)
        return BPETokenizer(vocab_size=vocab_size)
    elif tokenizer_type.lower() == 'rubert':
        model_name = kwargs.get('model_name', 'DeepPavlov/rubert-base-cased')
        return RuBERTTokenizer(model_name=model_name)
    else:
        raise ValueError(f"Неизвестный тип токенизатора: {tokenizer_type}. Используйте 'bpe' или 'rubert'")

