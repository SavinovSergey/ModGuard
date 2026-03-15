"""
Сервис классификации токсичности (двухстадийный пайплайн).

Stage 1 — regex pre-filter: мгновенно ловит очевидную токсичность.
Stage 2 — ML-модель (bert / rnn / tfidf / fasttext): классифицирует
          тексты, которые regex не поймал.

Regex также остаётся последним fallback в цепочке ModelManager на случай,
если все ML-модели недоступны.
"""
import logging
from typing import List, Dict, Any, Optional

from app.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

_EMPTY: Dict[str, Any] = {
    "is_toxic": False,
    "toxicity_score": 0.0,
    "toxicity_types": {},
    "tox_model_used": None,
}


class ToxicityService:
    """Классификация токсичности: regex pre-filter → ML fallback."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

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
    ) -> Dict[str, Any]:
        if not text:
            return dict(_EMPTY)

        # Stage 1: regex — быстрая проверка на очевидную токсичность
        regex_hit = self._regex_prefilter(text)
        if regex_hit:
            return regex_hit

        # Stage 2: ML-модель с fallback-цепочкой
        try:
            return self.model_manager.predict_with_fallback(text, preferred_model)
        except Exception as e:
            logger.error("Toxicity classification failed: %s", e)
            return dict(_EMPTY)

    def classify_batch(
        self,
        texts: List[str],
        preferred_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not texts:
            return []

        n = len(texts)
        results: List[Dict[str, Any]] = [dict(_EMPTY) for _ in range(n)]

        # --- Phase 1: regex pre-filter ---
        miss_indices: List[int] = []
        miss_texts: List[str] = []
        for i, text in enumerate(texts):
            if not text:
                continue
            miss_indices.append(i)
            miss_texts.append(text)

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

            for j, (idx, text) in enumerate(zip(miss_indices, miss_texts)):
                r = regex_results[j] if regex_results and j < len(regex_results) and regex_results[j] is not None else {}
                if r.get("is_toxic"):
                    r = dict(r)
                    r["tox_model_used"] = "regex"
                    results[idx] = r
                else:
                    ml_indices.append(idx)
                    ml_texts.append(text)
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
        except Exception as e:
            logger.error("Toxicity batch (ML) failed: %s", e)
            for idx, t in zip(ml_indices, ml_texts):
                results[idx] = self.classify(t, preferred_model=preferred_model)

        return results
