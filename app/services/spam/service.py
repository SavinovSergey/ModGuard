"""
Сервис классификации спама (двухстадийный пайплайн).

Stage 1 — regex pre-filter: мгновенно ловит очевидный спам.
Stage 2 — TF-IDF модель: классифицирует тексты, которые regex не поймал.

При ошибке любой стадии возвращается безопасный результат is_spam=False.
"""
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.spam.regex_model import SpamRegexModel
    from app.models.spam.tfidf_model import SpamTfidfModel

logger = logging.getLogger(__name__)

_EMPTY: Dict[str, Any] = {"is_spam": False, "spam_score": 0.0}


class SpamService:
    """Классификация спама: regex pre-filter -> TF-IDF fallback."""

    def __init__(
        self,
        spam_model: Optional["SpamTfidfModel"] = None,
        spam_regex_model: Optional["SpamRegexModel"] = None,
    ) -> None:
        self._spam_model = spam_model
        self._regex_model = spam_regex_model

    # ------------------------------------------------------------------
    # Regex pre-filter
    # ------------------------------------------------------------------

    def _regex_prefilter(self, text: str) -> Optional[Dict[str, Any]]:
        """Возвращает результат, если regex нашёл спам, иначе None."""
        if self._regex_model is None or not self._regex_model.is_loaded:
            return None
        try:
            result = self._regex_model.predict(text)
        except Exception as e:
            logger.debug("Spam regex pre-filter error: %s", e)
            return None
        if result.get("is_spam"):
            return result
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> Dict[str, Any]:
        if not text:
            return dict(_EMPTY)

        # Stage 1: regex
        regex_hit = self._regex_prefilter(text)
        if regex_hit:
            return regex_hit

        # Stage 2: TF-IDF
        if self._spam_model and self._spam_model.is_loaded:
            try:
                return self._spam_model.predict(text)
            except Exception as e:
                logger.warning("Spam classify failed: %s", e)

        return dict(_EMPTY)

    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []

        n = len(texts)
        results: List[Dict[str, Any]] = [dict(_EMPTY) for _ in range(n)]

        # --- Phase 1: regex pre-filter ---
        ml_indices: List[int] = []
        ml_texts: List[str] = []

        if self._regex_model and self._regex_model.is_loaded:
            try:
                regex_results = self._regex_model.predict_batch(texts)
            except Exception as e:
                logger.debug("Spam regex pre-filter batch error: %s", e)
                regex_results = [None] * n

            for i, text in enumerate(texts):
                if not text:
                    continue
                r = regex_results[i] if regex_results[i] is not None else {}
                if r.get("is_spam"):
                    results[i] = r
                else:
                    ml_indices.append(i)
                    ml_texts.append(text)
        else:
            for i, text in enumerate(texts):
                if text:
                    ml_indices.append(i)
                    ml_texts.append(text)

        if not ml_texts:
            return results

        # --- Phase 2: TF-IDF ---
        if self._spam_model and self._spam_model.is_loaded:
            try:
                batch_results = self._spam_model.predict_batch(ml_texts)
                for j, idx in enumerate(ml_indices):
                    results[idx] = batch_results[j]
            except Exception as e:
                logger.warning("Spam batch (TF-IDF) failed: %s", e)

        return results
