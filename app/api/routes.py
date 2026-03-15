"""API: только постановка задач в очередь и выдача результата по task_id. Классификация в backend."""
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Request

from app.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifySubmitResponse,
    BatchAsyncRequest,
    BatchAsyncResponse,
    TaskStatusResponse,
    HealthResponse,
    StatsResponse,
)
from app.core.config import settings
from app.core.task_store import TaskStore
from app.core.cache import ModerationCache, NoOpModerationCache
from app.core.db import create_task_pg, get_task_pg, set_task_result_pg
from app.core.queue import publish_task_request, publish_task_result
from app import __version__

logger = logging.getLogger(__name__)

router = APIRouter()


def get_task_store() -> TaskStore:
    from app.main import task_store
    return task_store


def get_moderation_cache(request: Request):
    """Кэш модерации (Redis или no-op при недоступности Redis)."""
    cache = getattr(request.app.state, "moderation_cache", None)
    return cache if cache is not None else NoOpModerationCache()


def _require_queue():
    """Требует наличия очереди и БД для постановки задач."""
    if not settings.rabbitmq_url or not settings.database_url:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: queue and database required (backend not configured)",
        )


def _get_cached_safe(cache, text: str):
    """Возвращает результат из кэша или None. При ошибке Redis — None (не нагружаем очередь лишними вызовами)."""
    if not text or not cache:
        return None
    try:
        return cache.get_cached_result(text)
    except Exception as e:
        logger.warning("Cache get failed: %s", e)
        return None


@router.post("/classify", response_model=ClassifySubmitResponse, tags=["classification"])
async def classify(request: ClassifyRequest, request_ctx: Request):
    """
    Ставит один текст в очередь на классификацию. Возвращает task_id.
    При попадании в кэш результат сразу пишется в БД и в очередь результатов (без очереди запросов).
    Результат получать по GET /tasks/{task_id}.
    """
    _require_queue()
    cache = get_moderation_cache(request_ctx)
    task_id = str(uuid.uuid4())
    items_payload = [{"id": "0", "text": request.text}]
    source = "web"

    cached = _get_cached_safe(cache, request.text)
    if cached is not None:
        create_task_pg(task_id, items_payload, source=source)
        set_task_result_pg(task_id, [cached], status="completed", from_cache=True)
        publish_task_result(task_id, source, [{"ref": {}, **cached}])
        return ClassifySubmitResponse(task_id=task_id)

    create_task_pg(task_id, items_payload, source=source)
    publish_task_request(
        task_id,
        [{"text": request.text, "ref": {}}],
        source=source,
        preferred_model=request.preferred_model,
    )
    return ClassifySubmitResponse(task_id=task_id)


@router.post(
    "/classify/batch-async",
    response_model=BatchAsyncResponse,
    tags=["classification"],
)
async def classify_batch_async(request: BatchAsyncRequest, request_ctx: Request):
    """
    Ставит батч в очередь, возвращает task_id (при частичном кэше — task_ids).
    При полном попадании в кэш — сразу в БД и в очередь результатов. При частичном — кэш-часть сразу в очередь результатов, остальное в очередь запросов (два task_id).
    """
    if len(request.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum ({settings.max_batch_size})",
        )
    _require_queue()
    cache = get_moderation_cache(request_ctx)
    source = request.source or "web"

    items_payload = [{"id": item.id or str(i), "text": item.text} for i, item in enumerate(request.items)]
    cached_results = []  # (index, result)
    miss_items = []  # (index, item with ref)
    for i, item in enumerate(request.items):
        ref = getattr(item, "ref", None) or {}
        c = _get_cached_safe(cache, item.text)
        if c is not None:
            cached_results.append((i, c, ref))
        else:
            miss_items.append((i, {"text": item.text, "ref": ref}))

    n_cached = len(cached_results)
    n_miss = len(miss_items)

    if n_cached == len(request.items):
        # Все в кэше — в БД и в очередь результатов, в очередь запросов не публикуем
        task_id = str(uuid.uuid4())
        by_index = {i: (c, ref) for i, c, ref in cached_results}
        results_for_db = [by_index[i][0] for i in range(len(request.items))]
        out = [{"ref": by_index[i][1], **by_index[i][0]} for i in range(len(request.items))]
        create_task_pg(task_id, items_payload, source=source, user_id=request.user_id)
        set_task_result_pg(task_id, results_for_db, status="completed", from_cache=True)
        publish_task_result(task_id, source, out)
        return BatchAsyncResponse(task_id=task_id)

    if n_miss == len(request.items):
        # Ничего в кэше — как раньше
        task_id = str(uuid.uuid4())
        items_for_queue = [{"text": item.text, "ref": {}} for item in request.items]
        create_task_pg(task_id, items_payload, source=source, user_id=request.user_id)
        publish_task_request(
            task_id,
            items_for_queue,
            source=request.source,
            preferred_model=request.preferred_model,
        )
        return BatchAsyncResponse(task_id=task_id)

    # Частичный кэш: кэш-часть — свой task_id, в очередь результатов; остальное — свой task_id, в очередь запросов
    task_id_cached = str(uuid.uuid4())
    task_id_miss = str(uuid.uuid4())
    payload_cached = [items_payload[i] for i, _, _ in cached_results]
    payload_miss = [items_payload[i] for i, _ in miss_items]
    results_cached = [c for _, c, _ in cached_results]
    create_task_pg(task_id_cached, payload_cached, source=source, user_id=request.user_id)
    set_task_result_pg(task_id_cached, results_cached, status="completed", from_cache=True)
    out_cached = [{"ref": ref, **c} for _, c, ref in cached_results]
    publish_task_result(task_id_cached, source, out_cached)

    items_miss_for_queue = [{"text": m["text"], "ref": m["ref"]} for _, m in miss_items]
    create_task_pg(task_id_miss, payload_miss, source=source, user_id=request.user_id)
    publish_task_request(
        task_id_miss,
        items_miss_for_queue,
        source=request.source,
        preferred_model=request.preferred_model,
    )
    return BatchAsyncResponse(task_id=task_id_miss, task_ids=[task_id_cached, task_id_miss])


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["classification"],
)
async def get_task_status(task_id: str, task_store: TaskStore = Depends(get_task_store)):
    """Получить статус и результат по task_id (polling). Читает из Postgres, иначе Redis/in-memory."""
    data = None
    if settings.database_url:
        data = get_task_pg(task_id)
    if data is None:
        data = task_store.get_task(task_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Task not found or expired")
    results_raw = data.get("results")
    results = [ClassifyResponse(**r) for r in results_raw] if results_raw is not None else None
    return TaskStatusResponse(
        status=data["status"],
        results=results,
        error=data.get("error"),
        user_id=data.get("user_id"),
    )


@router.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Проверка доступности API (очередь и бэкенд не проверяются)."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_info={"note": "Classification runs in backend worker"},
    )


@router.get("/stats", response_model=StatsResponse, tags=["monitoring"])
async def get_stats():
    """API не ведёт статистику моделей (классификация в backend)."""
    return StatsResponse(model_stats={}, current_model=None)
