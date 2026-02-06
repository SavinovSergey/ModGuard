"""Pydantic схемы для API"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Запрос на классификацию одного комментария"""
    text: str = Field(..., description="Текст комментария для классификации")
    context: Optional[List[str]] = Field(
        None, 
        description="Контекст обсуждения (предыдущие сообщения)"
    )
    preferred_model: Optional[str] = Field(
        None,
        description="Предпочтительная модель для использования"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Этот комментарий содержит нецензурную лексику",
                "context": None,
                "preferred_model": "regex"
            }
        }


class ToxicityType(BaseModel):
    """Тип токсичности с вероятностью"""
    name: str
    score: float = Field(..., ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    """Ответ на запрос классификации"""
    is_toxic: bool = Field(..., description="Токсичен ли комментарий")
    toxicity_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Вероятность токсичности (0-1)"
    )
    toxicity_types: Dict[str, float] = Field(
        default_factory=dict,
        description="Типы токсичности с вероятностями"
    )
    model_used: Optional[str] = Field(
        None,
        description="Модель, использованная для классификации"
    )
    error: Optional[str] = Field(
        None,
        description="Ошибка при классификации (если была)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "is_toxic": True,
                "toxicity_score": 0.95,
                "toxicity_types": {
                    "ебать": 1.0,
                    "прочее": 0.8
                },
                "model_used": "regex"
            }
        }


class BatchClassifyRequest(BaseModel):
    """Запрос на batch-классификацию"""
    texts: List[str] = Field(
        ..., 
        min_length=1,
        max_length=1000,
        description="Список текстов для классификации (максимум 1000)"
    )
    preferred_model: Optional[str] = Field(
        None,
        description="Предпочтительная модель для использования"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Это нормальный комментарий",
                    "Это токсичный комментарий"
                ],
                "preferred_model": "regex"
            }
        }


class BatchClassifyResponse(BaseModel):
    """Ответ на batch-запрос"""
    results: List[ClassifyResponse] = Field(
        ...,
        description="Результаты классификации для каждого текста"
    )
    total: int = Field(..., description="Общее количество обработанных текстов")
    model_used: Optional[str] = Field(
        None,
        description="Модель, использованная для классификации"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "is_toxic": False,
                        "toxicity_score": 0.0,
                        "toxicity_types": {}
                    },
                    {
                        "is_toxic": True,
                        "toxicity_score": 0.95,
                        "toxicity_types": {"ебать": 1.0}
                    }
                ],
                "total": 2,
                "model_used": "regex"
            }
        }


class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str = Field(..., description="Статус сервиса")
    version: str = Field(..., description="Версия сервиса")
    model_info: Optional[Dict] = Field(None, description="Информация о модели")


class StatsResponse(BaseModel):
    """Статистика использования моделей"""
    model_stats: Dict[str, Dict[str, int]] = Field(
        ...,
        description="Статистика по каждой модели"
    )
    current_model: Optional[str] = Field(None, description="Текущая активная модель")



