"""API routes для микросервиса"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    HealthResponse,
    StatsResponse
)
from app.services.classification import ClassificationService
from app.core.config import settings
from app import __version__

logger = logging.getLogger(__name__)

router = APIRouter()


def get_classification_service() -> ClassificationService:
    """Dependency для получения ClassificationService"""
    from app.main import classification_service
    return classification_service


def get_model_manager():
    """Dependency для получения ModelManager"""
    from app.main import model_manager
    return model_manager


@router.post("/classify", response_model=ClassifyResponse, tags=["classification"])
async def classify(
    request: ClassifyRequest,
    service: ClassificationService = Depends(get_classification_service)
):
    """
    Классифицирует один комментарий на токсичность
    
    - **text**: Текст комментария для классификации
    - **context**: Контекст обсуждения (опционально)
    - **preferred_model**: Предпочтительная модель (опционально)
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
    Классифицирует батч комментариев на токсичность
    
    - **texts**: Список текстов для классификации (максимум 1000)
    - **preferred_model**: Предпочтительная модель (опционально)
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
    Проверка здоровья сервиса
    
    Возвращает статус сервиса и информацию о загруженной модели
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


@router.get("/stats", response_model=StatsResponse, tags=["monitoring"])
async def get_stats(
    service: ClassificationService = Depends(get_classification_service)
):
    """
    Получить статистику использования моделей
    
    Возвращает статистику ошибок, таймаутов и успешных запросов для каждой модели
    """
    try:
        model_manager = get_model_manager()
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

