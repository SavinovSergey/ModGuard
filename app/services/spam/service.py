"""Сервис классификации спама (TF-IDF при наличии models/spam/, иначе заглушка)."""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.spam_tfidf_model import SpamTfidfModel

logger = logging.getLogger(__name__)


class SpamService:
    """
    Классификация спама. При переданной модели (SpamTfidfModel) — реальная инференция,
    иначе заглушка: is_spam=False, spam_score=0.0.
    """

    def __init__(self, spam_model: Optional["SpamTfidfModel"] = None) -> None:
        self._spam_model = spam_model

    def classify(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"is_spam": False, "spam_score": 0.0}
        if self._spam_model and self._spam_model.is_loaded:
            try:
                return self._spam_model.predict(text)
            except Exception as e:
                logger.warning("Spam classify failed: %s", e)
        return {"is_spam": False, "spam_score": 0.0}

    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        if self._spam_model and self._spam_model.is_loaded:
            try:
                return self._spam_model.predict_batch(texts)
            except Exception as e:
                logger.warning("Spam batch failed: %s", e)
        return [{"is_spam": False, "spam_score": 0.0} for _ in texts]
