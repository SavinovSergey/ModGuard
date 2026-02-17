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
    model_type: str = "fasttext"  # regex, tfidf, fasttext, rnn, rubert
    model_path: Optional[str] = None
    fallback_chain: List[str] = ["rubert", "rnn", "tfidf", "regex"]
    
    # Таймауты для моделей (в секундах)
    model_timeouts: dict = {
        "rubert": 0.1,
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


# Глобальный экземпляр настроек
settings = Settings()




