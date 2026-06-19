"""Конфигурация приложения"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from typing import Optional, List, Literal


class Settings(BaseSettings):
    """Настройки приложения"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra='ignore'  # Игнорировать дополнительные поля из .env, которых нет в модели
    )
    
    # Модель
    model_type: str = "tfidf"  # regex, tfidf, fasttext, rnn, bert
    model_path: Optional[str] = None
    fallback_chain: List[str] = ["tfidf", "bert", "rnn", "regex"]
    
    # BERT (опционально): если локальная директория моделей отсутствует,
    # можно грузить модель из HuggingFace по model-id (уменьшает размер репозитория).
    bert_hf_model_name: Optional[str] = "SergeySavinov/rubert-tiny-toxicity"

    # RNN (опционально): аналогично BERT, fallback на HuggingFace model-id.
    rnn_hf_model_name: Optional[str] = "SergeySavinov/rurnn-toxicity"
    
    # Таймауты для моделей (в секундах)
    model_timeouts: dict = {
        "bert": 0.1,
        "rnn": 0.15,
        "tfidf": 0.05,
        "regex": 0.01
    }
    
    # API
    api_version: str = "v1"
    api_prefix: str = "/api/v1"
    api_title: str = "ModGuard API"
    api_description: str = "API для классификации токсичности комментариев"
    
    # Производительность
    max_batch_size: int = 3000
    workers: int = 4
    # both — tox и spam параллельно; tox_only / spam_only — для бенчмарков (вторая ветка — заглушка)
    moderation_pipeline: Literal["both", "tox_only", "spam_only"] = "both"
    # Число процессов в каждом пуле (tox / spam) воркера. >1 — параллельный инференс
    # нескольких батчей внутри контейнера (по одному ядру на процесс).
    moderation_pool_workers: int = 1

    @field_validator("moderation_pipeline", mode="before")
    @classmethod
    def _normalize_moderation_pipeline(cls, value):
        if value is None:
            return "both"
        normalized = str(value).strip().lower()
        if normalized in ("both", "tox_only", "spam_only"):
            return normalized
        raise ValueError(
            "moderation_pipeline must be one of: both, tox_only, spam_only"
        )
    
    # Логирование
    log_level: str = "INFO"
    log_format: str = "json"  # json или text
    # Поэтапные метки времени цепочки (chain_timing) — для профилирования через validate_chain
    chain_timing: bool = False
    
    # Сервер
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # Redis (при None кэш и task_store работают in-memory / no-op)
    redis_url: Optional[str] = None
    cache_ttl_seconds: int = 3600
    cache_ttl_regex_seconds: int = 300
    task_result_ttl_seconds: int = 86400

    # Postgres (при None задачи хранятся в Redis/in-memory)
    database_url: Optional[str] = None

    # RabbitMQ (при None batch-async обрабатывается через BackgroundTasks, без очереди)
    rabbitmq_url: Optional[str] = None
    rabbitmq_queue_requests: str = "moderation.requests"
    rabbitmq_queue_results: str = "moderation.results"

    # Telegram (для listener и actions)
    telegram_bot_token: Optional[str] = None


# Глобальный экземпляр настроек
settings = Settings()




