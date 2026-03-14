"""
Сервис классификации токсичности (двухстадийный пайплайн).

Stage 1 — regex pre-filter: мгновенно ловит очевидную токсичность.
Stage 2 — ML-модель (bert / rnn / tfidf / fasttext): классифицирует
          тексты, которые regex не поймал.

Regex также остаётся последним fallback в цепочке ModelManager на случай,
если все ML-модели недоступны.
"""
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from app.core.model_manager import ModelManager

if TYPE_CHECKING:
    from app.core.cache import ModerationCache

logger = logging.getLogger(__name__)

_EMPTY: Dict[str, Any] = {
    "is_toxic": False,
    "toxicity_score": 0.0,
    "toxicity_types": {},
    "tox_model_used": None,
}


class ToxicityService:
    """Классификация токсичности: regex pre-filter → ML fallback."""

    def __init__(
        self,
        model_manager: ModelManager,
        moderation_cache: Optional["ModerationCache"] = None,
    ):
        self.model_manager = model_manager
        self._cache = moderation_cache

    # ------------------------------------------------------------------
    # Regex pre-filter
    # ------------------------------------------------------------------

    def _regex_prefilter(self, text: str) -> Optional[Dict[str, Any]]:
        """Возвращает результат, если regex нашёл токсичность, иначе None."""
        regex_model = self.model_manager.models.get("regex")
        if regex_model is None or not getattr(regex_model, "is_loaded", False):
            logger.debug(
                "Toxicity regex pre-filter skipped (single): regex_model=%s is_loaded=%s",
                regex_model is not None,
                getattr(regex_model, "is_loaded", False) if regex_model else False,
            )
            return None
        try:
            result = regex_model.predict(text)
        except Exception as e:
            logger.debug("Regex pre-filter error: %s", e)
            return None
        if result.get("is_toxic"):
            result = dict(result)
            result["tox_model_used"] = "regex"
            logger.info("Toxicity prediction by model 'regex' (pre-filter)")
            return result
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        preferred_model: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        if not text:
            return dict(_EMPTY)

        if use_cache and self._cache:
            cached = self._cache.get_cached_result(text)
            if cached is not None:
                return cached

        # Stage 1: regex — быстрая проверка на очевидную токсичность
        regex_hit = self._regex_prefilter(text)
        if regex_hit:
            if use_cache and self._cache:
                self._cache.set_cached_result(text, regex_hit, tox_model_used="regex")
            return regex_hit

        # Stage 2: ML-модель с fallback-цепочкой
        try:
            result = self.model_manager.predict_with_fallback(text, preferred_model)
            if use_cache and self._cache and result.get("tox_model_used") is not None:
                self._cache.set_cached_result(
                    text, result, tox_model_used=result.get("tox_model_used")
                )
            return result
        except Exception as e:
            logger.error("Toxicity classification failed: %s", e)
            return dict(_EMPTY)

    def classify_batch(
        self,
        texts: List[str],
        preferred_model: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        if not texts:
            return []

        n = len(texts)
        results: List[Dict[str, Any]] = [dict(_EMPTY) for _ in range(n)]

        # --- Phase 0: кэш ---
        miss_indices: List[int] = []
        miss_texts: List[str] = []
        for i, text in enumerate(texts):
            if not text:
                continue
            if use_cache and self._cache:
                cached = self._cache.get_cached_result(text)
                if cached is not None:
                    results[i] = cached
                    continue
            miss_indices.append(i)
            miss_texts.append(text)

        if not miss_texts:
            return results

        # --- Phase 1: regex pre-filter ---
        ml_indices: List[int] = []
        ml_texts: List[str] = []

        regex_model = self.model_manager.models.get("regex")
        if not regex_model or not getattr(regex_model, "is_loaded", False):
            logger.warning(
                "Toxicity regex pre-filter skipped: regex_model=%s is_loaded=%s",
                regex_model is not None,
                getattr(regex_model, "is_loaded", False) if regex_model else False,
            )
        if regex_model and regex_model.is_loaded:
            try:
                regex_results = regex_model.predict_batch(miss_texts)
            except Exception as e:
                logger.debug("Regex pre-filter batch error: %s", e)
                regex_results = [None] * len(miss_texts)

            regex_hits = 0
            for j, (idx, text) in enumerate(zip(miss_indices, miss_texts)):
                r = regex_results[j] if regex_results[j] is not None else {}
                if r.get("is_toxic"):
                    r = dict(r)
                    r["tox_model_used"] = "regex"
                    results[idx] = r
                    regex_hits += 1
                    if use_cache and self._cache:
                        self._cache.set_cached_result(text, r, tox_model_used="regex")
                else:
                    ml_indices.append(idx)
                    ml_texts.append(text)
            if regex_hits:
                logger.info(
                    "Toxicity batch: regex pre-filter matched %d of %d texts",
                    regex_hits,
                    len(miss_texts),
                )
        else:
            ml_indices = miss_indices
            ml_texts = miss_texts

        if not ml_texts:
            return results

        # --- Phase 2: ML-модель ---
        model = self.model_manager.get_model_with_fallback(preferred_model)
        try:
            batch_results = model.predict_batch(ml_texts)
            model_name = getattr(model, "model_name", None)
            for j, r in enumerate(batch_results):
                r = dict(r)
                r["tox_model_used"] = r.get("tox_model_used") or model_name
                results[ml_indices[j]] = r
                if use_cache and self._cache and j < len(ml_texts):
                    self._cache.set_cached_result(
                        ml_texts[j], r, tox_model_used=r.get("tox_model_used")
                    )
        except Exception as e:
            logger.error("Toxicity batch (ML) failed: %s", e)
            for idx, t in zip(ml_indices, ml_texts):
                results[idx] = self.classify(
                    t, preferred_model=preferred_model, use_cache=use_cache
                )

        return results
