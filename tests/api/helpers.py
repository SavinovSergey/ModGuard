"""Общие хелперы для API-тестов (async routes, кэш, очередь)."""
import json
from contextlib import contextmanager
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI

from app.api.routes import get_task_store, router
from app.core.cache import ModerationCache, _cache_key, _serialize_cache_value
from app.core.config import settings
from app.core.task_store import TaskStore


CACHED_RESULT: Dict[str, Any] = {
    "is_toxic": False,
    "toxicity_score": 0.1,
    "toxicity_types": {},
    "tox_model_used": "tfidf",
    "spam_model_used": "tfidf",
    "is_spam": False,
    "spam_score": 0.05,
}

CACHED_RESULT_TOXIC: Dict[str, Any] = {
    "is_toxic": True,
    "toxicity_score": 0.95,
    "toxicity_types": {"хуй": 1.0},
    "tox_model_used": "regex",
    "spam_model_used": None,
    "is_spam": False,
    "spam_score": 0.0,
}


class FakeAsyncRedis:
    """Минимальный async Redis для ModerationCache / TaskStore."""

    def __init__(self, data: Optional[Dict[str, str]] = None):
        self.data: Dict[str, str] = dict(data or {})
        self.get_calls: List[str] = []
        self.mget_calls: List[List[str]] = []

    async def get(self, key: str) -> Optional[str]:
        self.get_calls.append(key)
        return self.data.get(key)

    async def mget(self, keys: List[str]) -> List[Optional[str]]:
        self.mget_calls.append(list(keys))
        return [self.data.get(k) for k in keys]


class BrokenAsyncCache(ModerationCache):
    """Кэш, который падает на async-чтении (проверка graceful fallback в routes)."""

    def __init__(self) -> None:
        super().__init__(redis_client=object())

    async def aget_cached_result(self, text: str):
        raise RuntimeError("redis unavailable")

    async def aget_cached_results_batch(self, texts: List[str]):
        raise RuntimeError("redis unavailable")


def cache_for_texts(text_to_result: Dict[str, Dict[str, Any]]) -> ModerationCache:
    """ModerationCache с предзаполненными значениями по исходным текстам."""
    store: Dict[str, str] = {}
    for text, result in text_to_result.items():
        _, payload = _serialize_cache_value(result)
        store[_cache_key(text)] = payload
    return ModerationCache(FakeAsyncRedis(store))


@asynccontextmanager
async def _noop_lifespan(_app: FastAPI):
    yield


def make_api_test_app(cache: Optional[ModerationCache] = None) -> FastAPI:
    app = FastAPI(lifespan=_noop_lifespan)
    app.include_router(router, prefix=settings.api_prefix)
    if cache is not None:
        app.state.moderation_cache = cache
    store = TaskStore(None)
    app.dependency_overrides[get_task_store] = lambda: store
    app.state._test_task_store = store  # для тестов fallback
    return app


@contextmanager
def patch_api_queue_settings(max_batch_size: int = 1000):
    """Патч settings и async-зависимостей Postgres/очереди в routes."""
    with patch("app.api.routes.settings") as mock_settings:
        mock_settings.rabbitmq_url = "amqp://test/"
        mock_settings.database_url = "postgresql://test/"
        mock_settings.max_batch_size = max_batch_size
        with patch("app.api.routes.create_task_pg", new=AsyncMock()) as create_task:
            with patch("app.api.routes.set_task_result_pg", new=AsyncMock()) as set_result:
                with patch("app.api.routes.publish_task_request", new=AsyncMock()) as publish_req:
                    with patch("app.api.routes.publish_task_result", new=AsyncMock()) as publish_res:
                        yield {
                            "settings": mock_settings,
                            "create_task_pg": create_task,
                            "set_task_result_pg": set_result,
                            "publish_task_request": publish_req,
                            "publish_task_result": publish_res,
                        }
