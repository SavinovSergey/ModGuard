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

_EMPTY: Dict[str, Any] = {"is_spam": False, "spam_score": 0.0, "spam_model_used": None}


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
            result = dict(result)
            result["spam_model_used"] = "regex"
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
                result = self._spam_model.predict(text)
                result = dict(result)
                result["spam_model_used"] = result.get("spam_model_used") or "tfidf"
                return result
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

            regex_hits = 0
            for i, text in enumerate(texts):
                if not text:
                    continue
                r = regex_results[i] if regex_results[i] is not None else {}
                if r.get("is_spam"):
                    r = dict(r)
                    r["spam_model_used"] = "regex"
                    results[i] = r
                    regex_hits += 1
                else:
                    ml_indices.append(i)
                    ml_texts.append(text)
            if regex_hits:
                logger.info(
                    "Spam batch: regex pre-filter matched %d of %d texts",
                    regex_hits,
                    len(texts),
                )
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
                    r = dict(batch_results[j])
                    r["spam_model_used"] = r.get("spam_model_used") or "tfidf"
                    results[idx] = r
                logger.info(
                    "Spam batch: TF-IDF classified %d texts",
                    len(ml_texts),
                )
            except Exception as e:
                logger.warning("Spam batch (TF-IDF) failed: %s", e)

        return results
