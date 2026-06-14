"""Кэш результатов модерации по тексту (Redis или no-op)."""
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

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


def _ttl_seconds(tox_model_used: Optional[str]) -> int:
    """TTL кэша: короткий для regex-токсичности, иначе стандартный."""
    if tox_model_used == "regex":
        return settings.cache_ttl_regex_seconds
    return settings.cache_ttl_seconds


def _parse_cached_raw(raw: str) -> Dict[str, Any]:
    data = json.loads(raw)
    out = {
        "is_toxic": data.get("is_toxic", False),
        "toxicity_score": float(data.get("toxicity_score", 0.0)),
        "toxicity_types": data.get("toxicity_types") or {},
        "tox_model_used": data.get("tox_model_used"),
        "spam_model_used": data.get("spam_model_used"),
    }
    if "is_spam" in data:
        out["is_spam"] = data.get("is_spam", False)
        out["spam_score"] = float(data.get("spam_score", 0.0))
    return out


def _serialize_cache_value(
    result: Dict[str, Any],
    tox_model_used: Optional[str] = None,
) -> tuple[int, str]:
    """TTL и JSON для SETEX."""
    tox = tox_model_used or result.get("tox_model_used")
    ttl = _ttl_seconds(tox)
    payload = {
        "is_toxic": result.get("is_toxic", False),
        "toxicity_score": result.get("toxicity_score", 0.0),
        "toxicity_types": result.get("toxicity_types") or {},
        "tox_model_used": tox,
        "spam_model_used": result.get("spam_model_used"),
    }
    if "is_spam" in result:
        payload["is_spam"] = result.get("is_spam", False)
        payload["spam_score"] = float(result.get("spam_score", 0.0))
    return ttl, json.dumps(payload, ensure_ascii=False)


async def create_async_moderation_cache(redis_url: Optional[str]) -> "ModerationCache":
    """Async Redis-кэш для API/worker или NoOp при недоступности."""
    if not redis_url:
        return NoOpModerationCache()
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(redis_url, decode_responses=True)
        await client.ping()
        return ModerationCache(client)
    except Exception as e:
        logger.warning("Redis async connection failed: %s, cache disabled", e)
        return NoOpModerationCache()


class ModerationCache:
    """Кэш результатов модерации. Sync — worker/listener; async — API (redis.asyncio)."""

    def __init__(self, redis_client: Optional[Any] = None):
        self._redis = redis_client

    def get_cached_result(self, text: str) -> Optional[Dict[str, Any]]:
        """Sync: один GET (worker, scripts)."""
        if not self._redis or not text:
            return None
        try:
            raw = self._redis.get(_cache_key(text))
            if raw is None:
                return None
            return _parse_cached_raw(raw)
        except Exception as e:
            logger.warning("Cache get error: %s", e)
            return None

    async def aget_cached_result(self, text: str) -> Optional[Dict[str, Any]]:
        """Async: один GET (API)."""
        if not self._redis or not text:
            return None
        try:
            raw = await self._redis.get(_cache_key(text))
            if raw is None:
                return None
            return _parse_cached_raw(raw)
        except Exception as e:
            logger.warning("Cache async get error: %s", e)
            return None

    async def aget_cached_results_batch(self, texts: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Async: MGET — один roundtrip на весь батч."""
        if not self._redis or not texts:
            return [None] * len(texts)
        try:
            keys = [_cache_key(t) for t in texts]
            raws = await self._redis.mget(keys)
            out: List[Optional[Dict[str, Any]]] = []
            for raw in raws:
                if raw is None:
                    out.append(None)
                else:
                    try:
                        out.append(_parse_cached_raw(raw))
                    except Exception:
                        out.append(None)
            return out
        except Exception as e:
            logger.warning("Cache async mget error: %s", e)
            return [None] * len(texts)

    def set_cached_result(
        self,
        text: str,
        result: Dict[str, Any],
        tox_model_used: Optional[str] = None,
    ) -> None:
        """Sync SET (listener/scripts)."""
        if not self._redis or not text:
            return
        try:
            ttl, value = _serialize_cache_value(result, tox_model_used)
            self._redis.setex(_cache_key(text), ttl, value)
        except Exception as e:
            logger.warning("Cache set error: %s", e)

    async def aset_cached_results_batch(
        self,
        items: List[tuple[str, Dict[str, Any]]],
    ) -> int:
        """Async pipeline SETEX — один roundtrip на батч (worker после classify_batch)."""
        if not self._redis or not items:
            return 0
        try:
            pipe = self._redis.pipeline(transaction=False)
            n = 0
            for text, result in items:
                if not text:
                    continue
                if not (result.get("tox_model_used") or result.get("spam_model_used")):
                    continue
                ttl, value = _serialize_cache_value(result)
                pipe.setex(_cache_key(text), ttl, value)
                n += 1
            if n:
                await pipe.execute()
            return n
        except Exception as e:
            logger.warning("Cache async pipeline set error: %s", e)
            return 0


class NoOpModerationCache(ModerationCache):
    """Заглушка кэша при отсутствии Redis."""

    def __init__(self) -> None:
        super().__init__(redis_client=None)

    async def aget_cached_result(self, text: str) -> Optional[Dict[str, Any]]:
        return None

    async def aget_cached_results_batch(self, texts: List[str]) -> List[Optional[Dict[str, Any]]]:
        return [None] * len(texts)

    async def aset_cached_results_batch(self, items: List[tuple[str, Dict[str, Any]]]) -> int:
        return 0
