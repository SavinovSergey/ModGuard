"""API: только постановка задач в очередь и выдача результата по task_id. Классификация в backend."""
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends

from app.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifySubmitResponse,
    BatchClassifyRequest,
    BatchAsyncRequest,
    BatchAsyncResponse,
    TaskStatusResponse,
    HealthResponse,
    StatsResponse,
)
from app.core.config import settings
from app.core.task_store import TaskStore
from app.core.db import create_task_pg, get_task_pg
from app.core.queue import publish_task_request
from app import __version__

logger = logging.getLogger(__name__)

router = APIRouter()


def get_task_store() -> TaskStore:
    from app.main import task_store
    return task_store


def _require_queue():
    """Требует наличия очереди и БД для постановки задач."""
    if not settings.rabbitmq_url or not settings.database_url:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: queue and database required (backend not configured)",
        )


@router.post("/classify", response_model=ClassifySubmitResponse, tags=["classification"])
async def classify(request: ClassifyRequest):
    """
    Ставит один текст в очередь на классификацию. Возвращает task_id.
    Результат получать по GET /tasks/{task_id}.
    """
    _require_queue()
    task_id = str(uuid.uuid4())
    items_payload = [{"id": "0", "text": request.text}]
    items_for_queue = [{"text": request.text, "ref": {}}]
    create_task_pg(task_id, items_payload, source="web")
    publish_task_request(
        task_id,
        items_for_queue,
        source="web",
        preferred_model=request.preferred_model,
    )
    return ClassifySubmitResponse(task_id=task_id)


@router.post("/classify/batch", response_model=ClassifySubmitResponse, tags=["classification"])
async def classify_batch(request: BatchClassifyRequest):
    """
    Ставит батч текстов в очередь. Возвращает task_id.
    Результат получать по GET /tasks/{task_id} (массив results в том же порядке).
    """
    _require_queue()
    if len(request.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum ({settings.max_batch_size})",
        )
    task_id = str(uuid.uuid4())
    items_payload = [{"id": str(i), "text": t} for i, t in enumerate(request.texts)]
    items_for_queue = [{"text": t, "ref": {}} for t in request.texts]
    create_task_pg(task_id, items_payload, source="web")
    publish_task_request(
        task_id,
        items_for_queue,
        source="web",
        preferred_model=request.preferred_model,
    )
    return ClassifySubmitResponse(task_id=task_id)


@router.post(
    "/classify/batch-async",
    response_model=BatchAsyncResponse,
    tags=["classification"],
)
async def classify_batch_async(request: BatchAsyncRequest):
    """
    Ставит батч в очередь, возвращает task_id. Результат — по GET /tasks/{task_id}.
    """
    if len(request.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum ({settings.max_batch_size})",
        )
    _require_queue()
    task_id = str(uuid.uuid4())
    items_payload = [{"id": item.id or str(i), "text": item.text} for i, item in enumerate(request.items)]
    items_for_queue = [{"text": item.text, "ref": {}} for item in request.items]
    create_task_pg(task_id, items_payload, source=request.source or "web", user_id=request.user_id)
    publish_task_request(
        task_id,
        items_for_queue,
        source=request.source,
        preferred_model=request.preferred_model,
    )
    return BatchAsyncResponse(task_id=task_id)


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
