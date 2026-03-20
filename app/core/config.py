"""Конфигурация приложения"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional, List


class Settings(BaseSettings):
    """Настройки приложения"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra='ignore'  # Игнорировать дополнительные поля из .env, которых нет в модели
    )
    
    # Модель
    model_type: str = "bert"  # regex, tfidf, fasttext, rnn, bert
    model_path: Optional[str] = None
    fallback_chain: List[str] = ["bert", "rnn", "tfidf", "regex"]
    
    # BERT (опционально): если локальная директория моделей отсутствует,
    # можно грузить модель из HuggingFace по model-id (уменьшает размер репозитория).
    bert_hf_model_name: Optional[str] = "SergeySavinov/rubert-tiny-toxicity"
    
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
    api_title: str = "ToxicFilter API"
    api_description: str = "API для классификации токсичности комментариев"
    
    # Производительность
    max_batch_size: int = 1000
    workers: int = 4
    
    # Логирование
    log_level: str = "INFO"
    log_format: str = "json"  # json или text
    
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




