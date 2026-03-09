"""API routes модерации: классификация на токсичность и спам."""
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends

from app.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    BatchAsyncRequest,
    BatchAsyncResponse,
    TaskStatusResponse,
    HealthResponse,
    StatsResponse,
)
from app.services.classification import ClassificationService
from app.core.config import settings
from app.core.task_store import TaskStore
from app.core.db import (
    create_task_pg,
    get_task_pg,
    set_task_processing_pg,
    set_task_result_pg,
    set_task_failed_pg,
)
from app.core.queue import publish_task_request
from app import __version__

logger = logging.getLogger(__name__)

router = APIRouter()


def _process_batch_async(
    task_id: str,
    texts: List[str],
    preferred_model: Optional[str],
    classification_service: Optional["ClassificationService"] = None,
    task_store: Optional[TaskStore] = None,
) -> None:
    """Фоновая обработка батча: классификация на токсичность и спам, запись результата в task_store.
    service/store передаются из роута (Depends), чтобы тесты могли подменять их без lifespan.
    """
    if classification_service is None or task_store is None:
        from app.main import classification_service as _svc, task_store as _store
        classification_service = classification_service or _svc
        task_store = task_store or _store
    try:
        task_store.set_task_processing(task_id)
        results = classification_service.classify_batch(
            texts,
            preferred_model=preferred_model,
        )
        task_store.set_task_result(task_id, results, status="completed")
    except Exception as e:
        logger.exception("Batch async processing failed: %s", e)
        task_store.set_task_failed(task_id, str(e))


def get_classification_service() -> ClassificationService:
    """Dependency для получения ClassificationService"""
    from app.main import classification_service
    return classification_service


def get_model_manager():
    """Dependency для получения ModelManager"""
    from app.main import model_manager
    return model_manager


def get_task_store() -> TaskStore:
    """Dependency для получения TaskStore"""
    from app.main import task_store
    return task_store


@router.post("/classify", response_model=ClassifyResponse, tags=["classification"])
async def classify(
    request: ClassifyRequest,
    service: ClassificationService = Depends(get_classification_service)
):
    """
    Классифицирует один комментарий на токсичность и спам.

    В ответе: is_toxic, toxicity_score, toxicity_types, model_used (токсичность);
    is_spam, spam_score (спам).

    - **text**: Текст комментария для классификации
    - **context**: Контекст обсуждения (опционально)
    - **preferred_model**: Предпочтительная модель токсичности (опционально)
    """
    try:
        result = service.classify(
            request.text,
            context=request.context,
            preferred_model=request.preferred_model
        )
        
        # Добавляем информацию о модели, если доступна
        model_info = service.get_model_info()
        result['model_used'] = model_info.get('name')
        
        return ClassifyResponse(**result)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify/batch", response_model=BatchClassifyResponse, tags=["classification"])
async def classify_batch(
    request: BatchClassifyRequest,
    service: ClassificationService = Depends(get_classification_service)
):
    """
    Классифицирует батч комментариев на токсичность и спам.

    В каждом элементе ответа: поля токсичности (is_toxic, toxicity_score, ...) и спама (is_spam, spam_score).

    - **texts**: Список текстов для классификации (максимум 1000)
    - **preferred_model**: Предпочтительная модель токсичности (опционально)
    """
    if len(request.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum ({settings.max_batch_size})"
        )
    
    try:
        results = service.classify_batch(
            request.texts,
            preferred_model=request.preferred_model
        )
        
        # Преобразуем в ClassifyResponse
        response_results = [ClassifyResponse(**r) for r in results]
        
        # Получаем информацию о модели
        model_info = service.get_model_info()
        model_used = model_info.get('name')
        
        return BatchClassifyResponse(
            results=response_results,
            total=len(response_results),
            model_used=model_used
        )
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check(
    service: ClassificationService = Depends(get_classification_service)
):
    """
    Проверка здоровья сервиса.

    Возвращает статус и информацию о загруженной модели токсичности (модели спама учитываются отдельно).
    """
    try:
        model_info = service.get_model_info()
        status = "healthy" if model_info.get('is_loaded', False) else "degraded"
        
        return HealthResponse(
            status=status,
            version=__version__,
            model_info=model_info
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            model_info={"error": str(e)}
        )


@router.post(
    "/classify/batch-async",
    response_model=BatchAsyncResponse,
    tags=["classification"],
)
async def classify_batch_async(
    request: BatchAsyncRequest,
    background_tasks: BackgroundTasks,
    classification_service: ClassificationService = Depends(get_classification_service),
    task_store: TaskStore = Depends(get_task_store),
):
    """
    Принимает батч текстов, возвращает task_id. Результат получать по GET /tasks/{task_id}.

    Классификация выполняется на токсичность и спам; в результатах — поля is_toxic, is_spam и т.д.
    При настроенных Postgres и RabbitMQ задача уходит в воркер; иначе обрабатывается в фоне (Phase 1).
    """
    if len(request.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum ({settings.max_batch_size})",
        )
    task_id = str(uuid.uuid4())
    items_payload = [{"id": item.id, "text": item.text} for item in request.items]
    texts = [item.text for item in request.items]

    if settings.database_url and settings.rabbitmq_url:
        create_task_pg(task_id, items_payload, source=request.source, user_id=request.user_id)
        publish_task_request(
            task_id,
            items_payload,
            source=request.source,
            preferred_model=request.preferred_model,
        )
        return BatchAsyncResponse(task_id=task_id)

    task_store.create_task(task_id, items_payload)
    background_tasks.add_task(
        _process_batch_async,
        task_id,
        texts,
        request.preferred_model,
        classification_service,
        task_store,
    )
    return BatchAsyncResponse(task_id=task_id)


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["classification"],
)
async def get_task_status(
    task_id: str,
    task_store: TaskStore = Depends(get_task_store),
):
    """Получить статус и результат задачи по task_id (polling). Результаты содержат токсичность и спам. Читает из Postgres при наличии, иначе из Redis/in-memory."""
    data = None
    if settings.database_url:
        data = get_task_pg(task_id)
    if data is None:
        data = task_store.get_task(task_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Task not found or expired")
    results = data.get("results")
    if results is not None:
        results = [ClassifyResponse(**r) for r in results]
    return TaskStatusResponse(
        status=data["status"],
        results=results,
        error=data.get("error"),
        user_id=data.get("user_id"),
    )


@router.get("/stats", response_model=StatsResponse, tags=["monitoring"])
async def get_stats(
    model_manager = Depends(get_model_manager)
):
    """
    Получить статистику использования моделей токсичности.

    Возвращает статистику ошибок, таймаутов и успешных запросов для каждой модели (спам в статистику не входит).
    """
    try:
        stats = model_manager.get_stats()
        current_model = model_manager.current_model
        current_model_name = current_model.model_name if current_model else None
        
        return StatsResponse(
            model_stats=stats,
            current_model=current_model_name
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

