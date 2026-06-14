"""Дочерние процессы модерации: отдельная загрузка моделей tox / spam (без GIL)."""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_toxicity_service = None
_spam_service = None


def init_toxicity_process_worker() -> None:
    """Initializer ProcessPoolExecutor: только toxicity (+ ModelManager)."""
    global _toxicity_service
    from app.core.model_manager import ModelManager
    from app.core.config import settings
    from app.loader import register_all_models
    from app.services.toxicity import ToxicityService

    model_manager = ModelManager()
    register_all_models(model_manager)
    _toxicity_service = ToxicityService(model_manager)
    logger.info(
        "Toxicity process worker ready (model_type=%s)",
        settings.model_type,
    )


def init_spam_process_worker() -> None:
    """Initializer ProcessPoolExecutor: только spam."""
    global _spam_service
    from app.loader import get_spam_model, get_spam_regex_model
    from app.services.spam import SpamService

    _spam_service = SpamService(
        spam_model=get_spam_model(),
        spam_regex_model=get_spam_regex_model(),
    )
    logger.info("Spam process worker ready")


def run_toxicity_batch(
    payload: Tuple[List[str], Optional[str]],
) -> Tuple[List[Dict[str, Any]], float]:
    texts, preferred_model = payload
    if not texts:
        return [], 0.0
    t0 = time.perf_counter()
    results = _toxicity_service.classify_batch(texts, preferred_model=preferred_model)
    return results, (time.perf_counter() - t0) * 1000


def run_spam_batch(texts: List[str]) -> Tuple[List[Dict[str, Any]], float]:
    if not texts:
        return [], 0.0
    t0 = time.perf_counter()
    results = _spam_service.classify_batch(texts)
    return results, (time.perf_counter() - t0) * 1000
