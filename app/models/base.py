"""Base abstractions for toxicity classification model wrappers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from app.preprocessing.text_processor import TextProcessor


class BaseToxicityModel(ABC):
    """Common contract and helpers for all toxicity model wrappers."""

    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.is_loaded = False

    @abstractmethod
    def load(self, model_path: str = None, **kwargs) -> None:
        """Load model weights/artifacts from disk or external source."""

    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict toxicity for a single text."""

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Fallback batch inference via sequential `predict` calls."""
        return [self.predict(text) for text in texts]

    def ensure_loaded(self) -> None:
        """Raise if model is not loaded yet."""
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена. Вызовите load() перед использованием.")

    @staticmethod
    def empty_result() -> Dict[str, Any]:
        """Unified empty/non-inferable result payload."""
        return {
            "is_toxic": False,
            "toxicity_score": 0.0,
            "toxicity_types": {},
        }

    def _base_info(self, description: str, version: str = "1.0.0") -> Dict[str, Any]:
        """Base info payload for `get_model_info` methods."""
        return {
            "name": self.model_name,
            "type": self.model_type,
            "is_loaded": self.is_loaded,
            "version": version,
            "description": description,
        }

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model info and runtime metadata."""


class ClassicalTextModelBase(BaseToxicityModel, ABC):
    """Base for regex/tfidf/fasttext models using full text processing."""

    def __init__(self, model_name: str, model_type: str):
        super().__init__(model_name=model_name, model_type=model_type)
        self.text_processor = TextProcessor()

    def preprocess_text(self, text: str) -> str:
        if text is None or not isinstance(text, str):
            return ""
        return self.text_processor.process(text)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return self.text_processor.process_batch(texts)

    @staticmethod
    def non_empty_indexes(texts: List[str]) -> List[int]:
        return [i for i, text in enumerate(texts) if text and text.strip()]


class NeuralTextModelBase(BaseToxicityModel, ABC):
    """Base for neural wrappers using normalization-only preprocessing."""

    def __init__(self, model_name: str, model_type: str, remove_punctuation: bool = True):
        super().__init__(model_name=model_name, model_type=model_type)
        self.remove_punctuation = bool(remove_punctuation)
        self._init_neural_text_processor()

    def _init_neural_text_processor(self) -> None:
        self.text_processor = TextProcessor(
            use_lemmatization=False,
            remove_stopwords=False,
            remove_punkt=self.remove_punctuation,
        )

    def set_remove_punctuation(self, value: bool) -> None:
        """Согласовать нормализацию с обучением (поле remove_punctuation в params.json)."""
        self.remove_punctuation = bool(value)
        self._init_neural_text_processor()

    def preprocess_text(self, text: str) -> str:
        if text is None or not isinstance(text, str):
            return ""
        return self.text_processor.normalize(text)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        return [self.preprocess_text(text) for text in texts]

    @staticmethod
    def non_empty_indexes(texts: List[str]) -> List[int]:
        return [i for i, text in enumerate(texts) if text and text.strip()]




