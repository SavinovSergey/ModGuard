"""Кэш результатов модерации по тексту (Redis или no-op)."""
import hashlib
import json
import logging
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

CACHE_KEY_PREFIX = "moderation:cache:"


def _normalize_text_for_cache(text: str) -> str:
    """Единая нормализация текста для ключа кэша (strip + lower)."""
    if not text or not isinstance(text, str):
        return ""
    return text.strip().lower()


def _cache_key(text: str) -> str:
    normalized = _normalize_text_for_cache(text)
    h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{CACHE_KEY_PREFIX}{h}"


def _ttl_seconds(model_used: Optional[str]) -> int:
    if model_used == "regex":
        return settings.cache_ttl_regex_seconds
    return settings.cache_ttl_seconds


class ModerationCache:
    """Кэш результатов модерации. При redis_url=None — no-op."""

    def __init__(self, redis_client: Optional[Any] = None):
        self._redis = redis_client

    def get_cached_result(self, text: str) -> Optional[Dict[str, Any]]:
        """Возвращает закэшированный результат по тексту или None."""
        if not self._redis or not text:
            return None
        try:
            key = _cache_key(text)
            raw = self._redis.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            out = {
                "is_toxic": data.get("is_toxic", False),
                "toxicity_score": float(data.get("toxicity_score", 0.0)),
                "toxicity_types": data.get("toxicity_types") or {},
                "model_used": data.get("model_used"),
            }
            if "is_spam" in data:
                out["is_spam"] = data.get("is_spam", False)
                out["spam_score"] = float(data.get("spam_score", 0.0))
            return out
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set_cached_result(
        self,
        text: str,
        result: Dict[str, Any],
        model_used: Optional[str] = None,
    ) -> None:
        """Сохраняет результат в кэш. TTL зависит от model_used (regex — короче)."""
        if not self._redis or not text:
            return
        model = model_used or result.get("model_used")
        ttl = _ttl_seconds(model)
        payload = {
            "is_toxic": result.get("is_toxic", False),
            "toxicity_score": result.get("toxicity_score", 0.0),
            "toxicity_types": result.get("toxicity_types") or {},
            "model_used": model,
        }
        if "is_spam" in result:
            payload["is_spam"] = result.get("is_spam", False)
            payload["spam_score"] = float(result.get("spam_score", 0.0))
        try:
            key = _cache_key(text)
            self._redis.setex(
                key,
                ttl,
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")


class NoOpModerationCache(ModerationCache):
    """Заглушка кэша при отсутствии Redis: get всегда None, set ничего не делает."""

    def __init__(self) -> None:
        super().__init__(redis_client=None)
