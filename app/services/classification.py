"""
Единый сервис модерации: токсичность + спам.
Токсичность и спам считаются параллельно; своя предобработка у каждого модуля.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.core.model_manager import ModelManager
from app.services.toxicity import ToxicityService
from app.services.spam import SpamService

if TYPE_CHECKING:
    from app.core.cache import ModerationCache
    from app.models.spam.regex_model import SpamRegexModel
    from app.models.spam.tfidf_model import SpamTfidfModel

logger = logging.getLogger(__name__)

_DEFAULT_SPAM = {"is_spam": False, "spam_score": 0.0}


def _ensure_spam_fields(r: Dict[str, Any]) -> Dict[str, Any]:
    """Добавляет поля спама и tox_model_used/spam_model_used в результат, если их нет."""
    out = dict(r)
    if "is_spam" not in out:
        out["is_spam"] = False
    if "spam_score" not in out:
        out["spam_score"] = 0.0
    if "tox_model_used" not in out:
        out["tox_model_used"] = None
    if "spam_model_used" not in out:
        out["spam_model_used"] = None
    return out


def _merge_toxicity_spam(tox: Dict[str, Any], spam: Dict[str, Any]) -> Dict[str, Any]:
    """Объединяет результат токсичности и спама в один словарь."""
    out = dict(tox)
    out["is_spam"] = spam.get("is_spam", False)
    out["spam_score"] = float(spam.get("spam_score", 0.0))
    out["tox_model_used"] = tox.get("tox_model_used")
    out["spam_model_used"] = spam.get("spam_model_used")
    return out


class ClassificationService:
    """Фасад модерации: токсичность и спам считаются параллельно."""

    def __init__(
        self,
        model_manager: ModelManager,
        moderation_cache: Optional["ModerationCache"] = None,
        spam_model: Optional["SpamTfidfModel"] = None,
        spam_regex_model: Optional["SpamRegexModel"] = None,
    ):
        self.model_manager = model_manager
        self._cache = moderation_cache
        self._toxicity = ToxicityService(model_manager, moderation_cache)
        self._spam = SpamService(spam_model=spam_model, spam_regex_model=spam_regex_model)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="moderation")

    def classify(
        self,
        text: str,
        context: Optional[List[str]] = None,
        preferred_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Классификация токсичности и спама (параллельно). При попадании в кэш — возврат из кэша."""
        if not text:
            return {
                "is_toxic": False,
                "toxicity_score": 0.0,
                "toxicity_types": {},
                "tox_model_used": None,
                "spam_model_used": None,
                "is_spam": False,
                "spam_score": 0.0,
            }
        if self._cache:
            cached = self._cache.get_cached_result(text)
            if cached is not None:
                return _ensure_spam_fields(cached)
        f_tox = self._executor.submit(
            self._toxicity.classify,
            text,
            preferred_model=preferred_model,
            use_cache=False,
        )
        f_spam = self._executor.submit(self._spam.classify, text)
        tox = f_tox.result()
        spam = f_spam.result()
        merged = _merge_toxicity_spam(tox, spam)
        if self._cache and (merged.get("tox_model_used") or merged.get("spam_model_used")):
            self._cache.set_cached_result(
                text, merged, tox_model_used=merged.get("tox_model_used")
            )
        return merged

    def classify_batch(
        self,
        texts: List[str],
        preferred_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Батч: токсичность и спам параллельно по всему батчу, затем объединение по индексу."""
        if not texts:
            return []
        n = len(texts)
        results: List[Dict[str, Any]] = [{} for _ in range(n)]
        miss_indices: List[int] = []
        miss_texts: List[str] = []
        for i, text in enumerate(texts):
            if not text:
                results[i] = {
                    "is_toxic": False,
                    "toxicity_score": 0.0,
                    "toxicity_types": {},
                    "tox_model_used": None,
                    "spam_model_used": None,
                    "is_spam": False,
                    "spam_score": 0.0,
                }
                continue
            if self._cache:
                cached = self._cache.get_cached_result(text)
                if cached is not None:
                    results[i] = _ensure_spam_fields(cached)
                    continue
            miss_indices.append(i)
            miss_texts.append(text)
        if not miss_texts:
            return results
        f_tox = self._executor.submit(
            self._toxicity.classify_batch,
            miss_texts,
            preferred_model=preferred_model,
            use_cache=False,
        )
        f_spam = self._executor.submit(self._spam.classify_batch, miss_texts)
        tox_list = f_tox.result()
        spam_list = f_spam.result()
        for k, idx in enumerate(miss_indices):
            tox = tox_list[k] if k < len(tox_list) else {}
            spam = spam_list[k] if k < len(spam_list) else _DEFAULT_SPAM
            merged = _merge_toxicity_spam(tox, spam)
            results[idx] = merged
            if self._cache and miss_texts[k]:
                self._cache.set_cached_result(
                    miss_texts[k], merged, tox_model_used=merged.get("tox_model_used")
                )
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Информация о текущей модели (токсичность)."""
        try:
            model = self.model_manager.get_current_model()
            return model.get_model_info()
        except RuntimeError:
            return {
                "name": "none",
                "type": "none",
                "is_loaded": False,
                "error": "No model loaded",
            }
        except Exception as e:
            return {
                "name": "none",
                "type": "none",
                "is_loaded": False,
                "error": str(e),
            }
