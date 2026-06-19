"""Unit-тесты async-кэша ModerationCache (MGET, pipeline SET, NoOp)."""
import json

import pytest

from app.core.cache import (
    ModerationCache,
    NoOpModerationCache,
    _cache_key,
    _normalize_text_for_cache,
    _parse_cached_raw,
    _serialize_cache_value,
    _ttl_seconds,
)
from tests.api.helpers import FakeAsyncRedis, CACHED_RESULT


class FakePipeline:
    def __init__(self, parent: FakeAsyncRedis):
        self._parent = parent
        self._ops: list[tuple[str, str, str]] = []

    def setex(self, key: str, ttl: int, value: str):
        self._ops.append(("setex", key, value))
        return self

    async def execute(self):
        for op, key, value in self._ops:
            if op == "setex":
                self._parent.data[key] = value
        return [True] * len(self._ops)


class FakeAsyncRedisWithPipeline(FakeAsyncRedis):
    def pipeline(self, transaction=False):
        return FakePipeline(self)


@pytest.mark.asyncio
async def test_aget_cached_result_hit_and_miss():
    text = "Привет, мир!"
    _, raw = _serialize_cache_value(CACHED_RESULT)
    redis = FakeAsyncRedis({_cache_key(text): raw})
    cache = ModerationCache(redis)

    hit = await cache.aget_cached_result(text)
    miss = await cache.aget_cached_result("другой текст")

    assert hit is not None
    assert hit["tox_model_used"] == "tfidf"
    assert hit["spam_score"] == pytest.approx(0.05)
    assert miss is None
    assert len(redis.get_calls) == 2
    assert redis.get_calls[0] == _cache_key(text)


@pytest.mark.asyncio
async def test_aget_cached_results_batch_partial_mget():
    t1, t2, t3 = "alpha", "beta", "gamma"
    _, raw = _serialize_cache_value(CACHED_RESULT)
    redis = FakeAsyncRedis({_cache_key(t1): raw, _cache_key(t3): raw})
    cache = ModerationCache(redis)

    out = await cache.aget_cached_results_batch([t1, t2, t3])

    assert len(out) == 3
    assert out[0] is not None
    assert out[1] is None
    assert out[2] is not None
    assert len(redis.mget_calls) == 1


@pytest.mark.asyncio
async def test_aget_cached_results_batch_mget_error_returns_none_list():
    class FailingRedis:
        async def mget(self, keys):
            raise ConnectionError("down")

    cache = ModerationCache(FailingRedis())
    out = await cache.aget_cached_results_batch(["a", "b"])
    assert out == [None, None]


@pytest.mark.asyncio
async def test_aget_cached_results_batch_empty_texts():
    cache = ModerationCache(FakeAsyncRedis())
    assert await cache.aget_cached_results_batch([]) == []


@pytest.mark.asyncio
async def test_aset_cached_results_batch_pipeline():
    redis = FakeAsyncRedisWithPipeline()
    cache = ModerationCache(redis)
    items = [
        ("текст 1", {**CACHED_RESULT, "tox_model_used": "tfidf"}),
        ("", CACHED_RESULT),
        ("текст 2", {"is_toxic": True, "toxicity_score": 0.9, "spam_model_used": "tfidf", "is_spam": True, "spam_score": 0.8}),
    ]
    n = await cache.aset_cached_results_batch(items)
    assert n == 2
    assert _cache_key("текст 1") in redis.data
    assert _cache_key("текст 2") in redis.data


@pytest.mark.asyncio
async def test_noop_cache_async_methods():
    cache = NoOpModerationCache()
    assert await cache.aget_cached_result("x") is None
    assert await cache.aget_cached_results_batch(["a", "b"]) == [None, None]
    assert await cache.aset_cached_results_batch([("a", CACHED_RESULT)]) == 0


def test_normalize_text_for_cache_strip_lower():
    assert _normalize_text_for_cache("  AbC  ") == "abc"
    assert _normalize_text_for_cache("") == ""


def test_parse_cached_raw_includes_spam_fields():
    raw = json.dumps(
        {
            "is_toxic": True,
            "toxicity_score": 0.5,
            "toxicity_types": {},
            "tox_model_used": "regex",
            "spam_model_used": "tfidf",
            "is_spam": True,
            "spam_score": 0.77,
        }
    )
    parsed = _parse_cached_raw(raw)
    assert parsed["is_spam"] is True
    assert parsed["spam_score"] == pytest.approx(0.77)


def test_ttl_seconds_regex_shorter():
    from app.core.config import settings

    regex_ttl = _ttl_seconds("regex")
    default_ttl = _ttl_seconds("tfidf")
    assert regex_ttl == settings.cache_ttl_regex_seconds
    assert default_ttl == settings.cache_ttl_seconds
