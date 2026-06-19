"""
Единый сервис модерации: токсичность + спам.
В режиме both — параллельно в двух процессах (toxicity / spam), без GIL.

MODERATION_PIPELINE (env): both | tox_only | spam_only — для бенчмарков вторая ветка
отдаёт безопасную заглушку без вызова модели.
"""
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from app.core.config import settings
from app.core.model_manager import ModelManager
from app.services.toxicity import ToxicityService
from app.services.spam import SpamService

if TYPE_CHECKING:
    from app.models.spam.regex_model import SpamRegexModel
    from app.models.spam.tfidf_model import SpamTfidfModel

logger = logging.getLogger(__name__)

_DEFAULT_SPAM = {"is_spam": False, "spam_score": 0.0}

_STUB_TOX: Dict[str, Any] = {
    "is_toxic": False,
    "toxicity_score": 0.0,
    "toxicity_types": {},
    "tox_model_used": None,
}

_STUB_SPAM: Dict[str, Any] = {
    "is_spam": False,
    "spam_score": 0.0,
    "spam_model_used": None,
}


def _stub_tox_batch(count: int) -> List[Dict[str, Any]]:
    return [dict(_STUB_TOX) for _ in range(count)]


def _stub_spam_batch(count: int) -> List[Dict[str, Any]]:
    return [dict(_STUB_SPAM) for _ in range(count)]


def _empty_merged_result() -> Dict[str, Any]:
    return _merge_toxicity_spam(dict(_STUB_TOX), dict(_STUB_SPAM))


def _pipeline_mode() -> str:
    return settings.moderation_pipeline


def _merge_toxicity_spam(tox: Dict[str, Any], spam: Dict[str, Any]) -> Dict[str, Any]:
    """Объединяет результат токсичности и спама в один словарь."""
    out = dict(tox)
    out["is_spam"] = spam.get("is_spam", False)
    out["spam_score"] = float(spam.get("spam_score", 0.0))
    out["tox_model_used"] = tox.get("tox_model_used")
    out["spam_model_used"] = spam.get("spam_model_used")
    return out


class ClassificationService:
    """Фасад модерации: tox и spam параллельно (process pool или threads для тестов)."""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        spam_model: Optional["SpamTfidfModel"] = None,
        spam_regex_model: Optional["SpamRegexModel"] = None,
        *,
        use_process_pool: Optional[bool] = None,
    ):
        # Process pool — только worker без переданных моделей (production).
        self._use_process_pool = (
            use_process_pool if use_process_pool is not None else model_manager is None
        )
        self._model_manager = model_manager
        self._toxicity: Optional[ToxicityService] = None
        self._spam: Optional[SpamService] = None
        if model_manager is not None:
            self._toxicity = ToxicityService(model_manager)
            self._spam = SpamService(spam_model=spam_model, spam_regex_model=spam_regex_model)

        self._thread_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="moderation")
        self._tox_pool: Optional[ProcessPoolExecutor] = None
        self._spam_pool: Optional[ProcessPoolExecutor] = None
        self._pools_ready = False

    def _ensure_inprocess_services(self) -> None:
        if self._toxicity is not None and self._spam is not None:
            return
        from app.loader import get_spam_model, get_spam_regex_model, register_all_models

        self._model_manager = ModelManager()
        register_all_models(self._model_manager)
        self._toxicity = ToxicityService(self._model_manager)
        self._spam = SpamService(
            spam_model=get_spam_model(),
            spam_regex_model=get_spam_regex_model(),
        )

    def _ensure_process_pools(self) -> None:
        if self._pools_ready:
            return
        from app.services.classification_process import (
            init_spam_process_worker,
            init_toxicity_process_worker,
            run_spam_batch,
            run_toxicity_batch,
        )

        n_workers = max(1, settings.moderation_pool_workers)
        logger.info(
            "Starting moderation process pools (toxicity + spam, workers=%d each)...",
            n_workers,
        )
        t0 = time.perf_counter()
        self._tox_pool = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_toxicity_process_worker,
        )
        self._spam_pool = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_spam_process_worker,
        )
        # Прогрев: дождаться initializer во всех процессах каждого пула.
        tox_warm = [self._tox_pool.submit(run_toxicity_batch, ([], None)) for _ in range(n_workers)]
        spam_warm = [self._spam_pool.submit(run_spam_batch, []) for _ in range(n_workers)]
        for f in tox_warm:
            f.result()
        for f in spam_warm:
            f.result()
        self._pools_ready = True
        logger.info(
            "Moderation process pools ready in %.1fs",
            time.perf_counter() - t0,
        )

    def _timed_toxicity_batch(
        self,
        texts: List[str],
        preferred_model: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], float]:
        self._ensure_inprocess_services()
        t0 = time.perf_counter()
        results = self._toxicity.classify_batch(texts, preferred_model=preferred_model)
        return results, (time.perf_counter() - t0) * 1000

    def _timed_spam_batch(self, texts: List[str]) -> Tuple[List[Dict[str, Any]], float]:
        self._ensure_inprocess_services()
        t0 = time.perf_counter()
        results = self._spam.classify_batch(texts)
        return results, (time.perf_counter() - t0) * 1000

    def _run_both_process_batch(
        self,
        texts: List[str],
        preferred_model: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]], float]:
        from app.services.classification_process import run_spam_batch, run_toxicity_batch

        self._ensure_process_pools()
        assert self._tox_pool is not None and self._spam_pool is not None

        f_tox = self._tox_pool.submit(run_toxicity_batch, (texts, preferred_model))
        f_spam = self._spam_pool.submit(run_spam_batch, texts)
        tox_list, tox_ms = f_tox.result()
        spam_list, spam_ms = f_spam.result()
        return tox_list, tox_ms, spam_list, spam_ms

    def _run_both_thread_batch(
        self,
        texts: List[str],
        preferred_model: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]], float]:
        f_tox = self._thread_executor.submit(
            self._timed_toxicity_batch,
            texts,
            preferred_model,
        )
        f_spam = self._thread_executor.submit(self._timed_spam_batch, texts)
        tox_list, tox_ms = f_tox.result()
        spam_list, spam_ms = f_spam.result()
        return tox_list, tox_ms, spam_list, spam_ms

    def classify(
        self,
        text: str,
        context: Optional[List[str]] = None,
        preferred_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Классификация токсичности и спама (параллельно или одна ветка + заглушка)."""
        if not text:
            return _empty_merged_result()

        mode = _pipeline_mode()
        if mode == "tox_only":
            self._ensure_inprocess_services()
            tox = self._toxicity.classify(text, preferred_model=preferred_model)
            return _merge_toxicity_spam(tox, dict(_STUB_SPAM))
        if mode == "spam_only":
            self._ensure_inprocess_services()
            spam = self._spam.classify(text)
            return _merge_toxicity_spam(dict(_STUB_TOX), spam)

        if self._use_process_pool:
            tox_list, _, spam_list, _ = self._run_both_process_batch(
                [text],
                preferred_model,
            )
            tox = tox_list[0] if tox_list else dict(_STUB_TOX)
            spam = spam_list[0] if spam_list else dict(_STUB_SPAM)
            return _merge_toxicity_spam(tox, spam)

        self._ensure_inprocess_services()
        f_tox = self._thread_executor.submit(
            self._toxicity.classify,
            text,
            preferred_model=preferred_model,
        )
        f_spam = self._thread_executor.submit(self._spam.classify, text)
        return _merge_toxicity_spam(f_tox.result(), f_spam.result())

    def classify_batch(
        self,
        texts: List[str],
        preferred_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Батч: токсичность и спам параллельно, затем объединение по индексу."""
        if not texts:
            return []
        n = len(texts)
        results: List[Dict[str, Any]] = [{} for _ in range(n)]
        miss_indices: List[int] = []
        miss_texts: List[str] = []
        for i, text in enumerate(texts):
            if not text:
                results[i] = _empty_merged_result()
                continue
            miss_indices.append(i)
            miss_texts.append(text)
        if not miss_texts:
            return results

        mode = _pipeline_mode()
        executor_label = "process" if self._use_process_pool and mode == "both" else "inprocess"
        t_wall0 = time.perf_counter()
        tox_ms = 0.0
        spam_ms = 0.0

        if mode == "tox_only":
            tox_list, tox_ms = self._timed_toxicity_batch(miss_texts, preferred_model)
            spam_list = _stub_spam_batch(len(miss_texts))
        elif mode == "spam_only":
            tox_list = _stub_tox_batch(len(miss_texts))
            spam_list, spam_ms = self._timed_spam_batch(miss_texts)
        elif self._use_process_pool:
            executor_label = "process"
            tox_list, tox_ms, spam_list, spam_ms = self._run_both_process_batch(
                miss_texts,
                preferred_model,
            )
        else:
            executor_label = "thread"
            tox_list, tox_ms, spam_list, spam_ms = self._run_both_thread_batch(
                miss_texts,
                preferred_model,
            )

        wall_ms = (time.perf_counter() - t_wall0) * 1000
        from app.core import chain_timing

        if chain_timing.enabled():
            chain_timing.mark(
                "worker", "classify_tox", "end", n_items=len(miss_texts), ms=tox_ms
            )
            chain_timing.mark(
                "worker", "classify_spam", "end", n_items=len(miss_texts), ms=spam_ms
            )
        logger.info(
            "Classify batch timing mode=%s executor=%s items=%d "
            "wall_ms=%.0f tox_ms=%.0f spam_ms=%.0f",
            mode,
            executor_label,
            len(miss_texts),
            wall_ms,
            tox_ms,
            spam_ms,
        )

        for k, idx in enumerate(miss_indices):
            tox = tox_list[k] if k < len(tox_list) else dict(_STUB_TOX)
            spam = spam_list[k] if k < len(spam_list) else dict(_STUB_SPAM)
            results[idx] = _merge_toxicity_spam(tox, spam)
        return results

    def shutdown(self) -> None:
        """Останавливает process/thread pools (вызывать при shutdown worker)."""
        if self._tox_pool is not None:
            self._tox_pool.shutdown(wait=True)
            self._tox_pool = None
        if self._spam_pool is not None:
            self._spam_pool.shutdown(wait=True)
            self._spam_pool = None
        self._pools_ready = False
        self._thread_executor.shutdown(wait=True)

    def get_model_info(self) -> Dict[str, Any]:
        """Информация о текущей модели (токсичность)."""
        try:
            self._ensure_inprocess_services()
            model = self._model_manager.get_current_model()
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
