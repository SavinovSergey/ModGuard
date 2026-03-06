"""Сервис классификации токсичности (иерархический модуль, при необходимости выделяется в отдельный микросервис)."""
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from app.core.model_manager import ModelManager

if TYPE_CHECKING:
    from app.core.cache import ModerationCache

logger = logging.getLogger(__name__)


class ToxicityService:
    """Классификация токсичности комментариев (модели + кэш)."""

    def __init__(
        self,
        model_manager: ModelManager,
        moderation_cache: Optional["ModerationCache"] = None,
    ):
        self.model_manager = model_manager
        self._cache = moderation_cache

    def classify(
        self,
        text: str,
        preferred_model: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        if not text:
            return {
                "is_toxic": False,
                "toxicity_score": 0.0,
                "toxicity_types": {},
            }
        if use_cache and self._cache:
            cached = self._cache.get_cached_result(text)
            if cached is not None:
                return cached
        try:
            result = self.model_manager.predict_with_fallback(text, preferred_model)
            if use_cache and self._cache and result.get("model_used") is not None:
                self._cache.set_cached_result(
                    text, result, model_used=result.get("model_used")
                )
            return result
        except Exception as e:
            logger.error(f"Toxicity classification failed: {e}")
            return {
                "is_toxic": False,
                "toxicity_score": 0.0,
                "toxicity_types": {},
                "error": str(e),
            }

    def classify_batch(
        self,
        texts: List[str],
        preferred_model: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        if not texts:
            return []
        empty: Dict[str, Any] = {
            "is_toxic": False,
            "toxicity_score": 0.0,
            "toxicity_types": {},
            "model_used": None,
        }
        results: List[Dict[str, Any]] = [dict(empty) for _ in range(len(texts))]
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
        model = self.model_manager.get_model_with_fallback(preferred_model)
        try:
            batch_results = model.predict_batch(miss_texts)
            model_name = getattr(model, "model_name", None)
            for j, r in enumerate(batch_results):
                r = dict(r)
                r["model_used"] = r.get("model_used") or model_name
                if use_cache and self._cache and j < len(miss_texts):
                    self._cache.set_cached_result(
                        miss_texts[j], r, model_used=r.get("model_used")
                    )
                results[miss_indices[j]] = r
        except Exception as e:
            logger.error(f"Toxicity batch failed: {e}")
            for idx, t in zip(miss_indices, miss_texts):
                results[idx] = self.classify(t, preferred_model=preferred_model, use_cache=use_cache)
        return results
