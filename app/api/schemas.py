"""Pydantic схемы для API"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict


class ClassifyRequest(BaseModel):
    """Запрос на классификацию одного комментария"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Этот комментарий содержит нецензурную лексику",
                "context": None,
                "preferred_model": "regex"
            }
        }
    )
    
    text: str = Field(..., description="Текст комментария для классификации")
    context: Optional[List[str]] = Field(
        None, 
        description="Контекст обсуждения (предыдущие сообщения)"
    )
    preferred_model: Optional[str] = Field(
        None,
        description="Предпочтительная модель для использования"
    )


class ToxicityType(BaseModel):
    """Тип токсичности с вероятностью"""
    name: str
    score: float = Field(..., ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    """Ответ на запрос классификации (токсичность + спам)"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_toxic": True,
                "toxicity_score": 0.95,
                "toxicity_types": {"ебать": 1.0, "прочее": 0.8},
                "tox_model_used": "regex",
                "spam_model_used": "regex",
                "is_spam": False,
                "spam_score": 0.0,
            }
        }
    )

    is_toxic: bool = Field(..., description="Токсичен ли комментарий")
    toxicity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Вероятность токсичности (0-1)",
    )
    toxicity_types: Dict[str, float] = Field(
        default_factory=dict,
        description="Типы токсичности с вероятностями",
    )
    tox_model_used: Optional[str] = Field(
        None,
        description="Модель, использованная для классификации токсичности",
    )
    spam_model_used: Optional[str] = Field(
        None,
        description="Модель, использованная для классификации спама",
    )
    is_spam: bool = Field(
        False,
        description="Является ли сообщение спамом",
    )
    spam_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Вероятность спама (0-1)",
    )
    error: Optional[str] = Field(
        None,
        description="Ошибка при классификации (если была)",
    )


class BatchClassifyRequest(BaseModel):
    """Запрос на batch-классификацию"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "Это нормальный комментарий",
                    "Это токсичный комментарий"
                ],
                "preferred_model": "regex"
            }
        }
    )
    
    texts: List[str] = Field(
        ..., 
        min_length=1,
        # max_length убран, так как проверка выполняется в коде endpoint
        description="Список текстов для классификации (максимум 1000)"
    )
    preferred_model: Optional[str] = Field(
        None,
        description="Предпочтительная модель для использования"
    )


class BatchClassifyResponse(BaseModel):
    """Ответ на batch-запрос"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "is_toxic": False,
                        "toxicity_score": 0.0,
                        "toxicity_types": {},
                        "tox_model_used": "regex",
                        "spam_model_used": None,
                    },
                    {
                        "is_toxic": True,
                        "toxicity_score": 0.95,
                        "toxicity_types": {"ебать": 1.0},
                        "tox_model_used": "regex",
                        "spam_model_used": "tfidf",
                    }
                ],
                "total": 2,
            }
        }
    )

    results: List[ClassifyResponse] = Field(
        ...,
        description="Результаты классификации для каждого текста",
    )
    total: int = Field(..., description="Общее количество обработанных текстов")


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


class BatchAsyncItem(BaseModel):
    """Элемент батча для асинхронной классификации"""
    id: Optional[str] = Field(None, description="Локальный id для сопоставления результата")
    text: str = Field(..., description="Текст для модерации")


class BatchAsyncRequest(BaseModel):
    """Запрос на асинхронную batch-классификацию"""
    items: List[BatchAsyncItem] = Field(
        ...,
        min_length=1,
        description="Список элементов (текст + опционально id)"
    )
    source: Optional[str] = Field(None, description="Источник запроса (web, telegram, …)")
    user_id: Optional[int] = Field(None, description="Идентификатор отправителя (например Telegram user id)")
    preferred_model: Optional[str] = Field(None, description="Предпочтительная модель")


class ClassifySubmitResponse(BaseModel):
    """Ответ при постановке задачи в очередь (одиночный или батч): только task_id для polling."""
    task_id: str = Field(..., description="Идентификатор задачи; результат получать по GET /tasks/{task_id}")


class BatchAsyncResponse(BaseModel):
    """Ответ на запрос batch-async: только task_id"""
    task_id: str = Field(..., description="Идентификатор задачи для polling")


class TaskStatusResponse(BaseModel):
    """Статус и результат задачи по task_id"""
    status: str = Field(
        ...,
        description="queued | processing | completed | failed"
    )
    results: Optional[List[ClassifyResponse]] = Field(
        None,
        description="Результаты (при status=completed), в том же порядке что items"
    )
    error: Optional[str] = Field(None, description="Сообщение об ошибке (при status=failed)")
    user_id: Optional[int] = Field(None, description="Идентификатор отправителя (если передан при создании задачи)")




