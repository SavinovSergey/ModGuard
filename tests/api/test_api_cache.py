"""Тесты async API: cache-hit / cache-miss / partial cache для classify и batch-async."""
import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.core.cache import NoOpModerationCache
from app.core.task_store import TASK_KEY_PREFIX

from app.core.cache import ModerationCache
from tests.api.helpers import (
    BrokenAsyncCache,
    CACHED_RESULT,
    CACHED_RESULT_TOXIC,
    FakeAsyncRedis,
    cache_for_texts,
    make_api_test_app,
    patch_api_queue_settings,
)


@pytest.fixture
def api_client_no_cache():
    app = make_api_test_app(cache=NoOpModerationCache())
    with TestClient(app) as client:
        yield client, app


@pytest.fixture
def api_client():
    app = make_api_test_app()
    with TestClient(app) as client:
        yield client, app


# --- POST /classify ---


def test_classify_cache_miss_publishes_to_request_queue(api_client):
    """Без кэша: create_task + publish_task_request, без fast-path result."""
    client, _ = api_client
    with patch_api_queue_settings() as mocks:
        resp = client.post("/api/v1/classify", json={"text": "новый текст"})
    assert resp.status_code == 200
    assert "task_id" in resp.json()
    mocks["create_task_pg"].assert_awaited_once()
    mocks["publish_task_request"].assert_awaited_once()
    mocks["set_task_result_pg"].assert_not_awaited()
    mocks["publish_task_result"].assert_not_awaited()


def test_classify_cache_hit_skips_request_queue(api_client):
    """Cache hit: результат сразу в БД и moderation.results, очередь запросов не используется."""
    text = "закэшированный комментарий"
    app = make_api_test_app(cache=cache_for_texts({text: CACHED_RESULT}))
    with TestClient(app) as client:
        with patch_api_queue_settings() as mocks:
            resp = client.post("/api/v1/classify", json={"text": text})
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    mocks["create_task_pg"].assert_awaited_once()
    mocks["set_task_result_pg"].assert_awaited_once()
    set_kwargs = mocks["set_task_result_pg"].await_args.kwargs
    assert set_kwargs.get("from_cache") is True
    assert set_kwargs.get("status") == "completed"
    mocks["publish_task_result"].assert_awaited_once()
    pub_args = mocks["publish_task_result"].await_args
    assert pub_args.args[0] == task_id
    assert pub_args.args[2][0]["is_toxic"] is False
    mocks["publish_task_request"].assert_not_awaited()


def test_classify_cache_error_falls_back_to_queue(api_client):
    """Ошибка Redis на GET не ломает запрос — идём в очередь."""
    app = make_api_test_app(cache=BrokenAsyncCache())
    with TestClient(app) as client:
        with patch_api_queue_settings() as mocks:
            resp = client.post("/api/v1/classify", json={"text": "любой текст"})
    assert resp.status_code == 200
    mocks["publish_task_request"].assert_awaited_once()
    mocks["publish_task_result"].assert_not_awaited()


# --- POST /classify/batch-async ---


def test_batch_async_all_miss_single_task_id(api_client_no_cache):
    """Все промахи кэша → один task_id, только publish_task_request."""
    client, _ = api_client_no_cache
    with patch_api_queue_settings() as mocks:
        resp = client.post(
            "/api/v1/classify/batch-async",
            json={"items": [{"text": "a"}, {"text": "b"}]},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("task_ids") is None
    mocks["create_task_pg"].assert_awaited_once()
    mocks["publish_task_request"].assert_awaited_once()
    mocks["publish_task_result"].assert_not_awaited()


def test_batch_async_full_cache_hit(api_client):
    """Полный cache hit → один task_id, set_task_result_pg(from_cache=True), publish_task_result."""
    texts = ["кэш один", "кэш два"]
    app = make_api_test_app(
        cache=cache_for_texts({texts[0]: CACHED_RESULT, texts[1]: CACHED_RESULT_TOXIC})
    )
    with TestClient(app) as client:
        with patch_api_queue_settings() as mocks:
            resp = client.post(
                "/api/v1/classify/batch-async",
                json={"items": [{"id": "1", "text": texts[0]}, {"id": "2", "text": texts[1]}]},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("task_ids") is None
    mocks["create_task_pg"].assert_awaited_once()
    mocks["set_task_result_pg"].assert_awaited_once()
    assert mocks["set_task_result_pg"].await_args.kwargs.get("from_cache") is True
    mocks["publish_task_result"].assert_awaited_once()
    published = mocks["publish_task_result"].await_args.args[2]
    assert len(published) == 2
    assert published[1]["is_toxic"] is True
    mocks["publish_task_request"].assert_not_awaited()


def test_batch_async_partial_cache_split_task_ids(api_client):
    """Частичный кэш → task_ids=[cached_id, miss_id], два create_task, result + request."""
    hit_text, miss_text = "в кэше", "не в кэше"
    app = make_api_test_app(cache=cache_for_texts({hit_text: CACHED_RESULT}))
    with TestClient(app) as client:
        with patch_api_queue_settings() as mocks:
            resp = client.post(
                "/api/v1/classify/batch-async",
                json={
                    "items": [
                        {"id": "hit", "text": hit_text},
                        {"id": "miss", "text": miss_text},
                    ],
                    "user_id": 42,
                },
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_ids"] is not None
    assert len(data["task_ids"]) == 2
    cached_task_id, miss_task_id = data["task_ids"]
    assert data["task_id"] == miss_task_id
    assert mocks["create_task_pg"].await_count == 2
    mocks["publish_task_result"].assert_awaited_once()
    assert mocks["publish_task_result"].await_args.args[0] == cached_task_id
    mocks["publish_task_request"].assert_awaited_once()
    req_items = mocks["publish_task_request"].await_args.args[1]
    assert len(req_items) == 1
    assert req_items[0]["text"] == miss_text
    # user_id пробрасывается в create_task_pg
    for call in mocks["create_task_pg"].await_args_list:
        assert call.kwargs.get("user_id") == 42


def test_batch_async_cache_error_treats_as_all_miss(api_client):
    """Ошибка MGET → все элементы в очередь, как при полном промахе."""
    app = make_api_test_app(cache=BrokenAsyncCache())
    with TestClient(app) as client:
        with patch_api_queue_settings() as mocks:
            resp = client.post(
                "/api/v1/classify/batch-async",
                json={"items": [{"text": "x"}, {"text": "y"}]},
            )
    assert resp.status_code == 200
    assert resp.json().get("task_ids") is None
    mocks["publish_task_request"].assert_awaited_once()
    mocks["publish_task_result"].assert_not_awaited()


def test_batch_async_uses_mget_once(api_client):
    """Батч читает кэш одним MGET (не N отдельных GET)."""
    redis = FakeAsyncRedis()
    cache = ModerationCache(redis)
    app = make_api_test_app(cache=cache)
    with TestClient(app) as client:
        with patch_api_queue_settings():
            client.post(
                "/api/v1/classify/batch-async",
                json={"items": [{"text": "t1"}, {"text": "t2"}, {"text": "t3"}]},
            )
    assert len(redis.mget_calls) == 1
    assert len(redis.mget_calls[0]) == 3


# --- GET /tasks async fallback ---


def test_get_task_reads_task_store_via_aget_task(api_client):
    """Если Postgres пуст, GET /tasks читает task_store.aget_task (async path)."""
    client, app = api_client
    task_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    payload = {
        "status": "completed",
        "results": [
            {
                "is_toxic": False,
                "toxicity_score": 0.0,
                "toxicity_types": {},
                "tox_model_used": "regex",
                "spam_model_used": None,
                "is_spam": False,
                "spam_score": 0.0,
            }
        ],
        "error": None,
    }
    redis = FakeAsyncRedis({f"{TASK_KEY_PREFIX}{task_id}": json.dumps(payload, ensure_ascii=False)})
    from app.core.task_store import TaskStore

    async_store = TaskStore(redis)
    app.dependency_overrides.clear()
    from app.api.routes import get_task_store

    app.dependency_overrides[get_task_store] = lambda: async_store

    with patch("app.api.routes.get_task_pg", new=AsyncMock(return_value=None)):
        with patch("app.api.routes.settings") as m:
            m.database_url = "postgresql://test/"
            resp = client.get(f"/api/v1/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    assert len(redis.get_calls) >= 1
