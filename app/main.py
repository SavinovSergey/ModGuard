"""Главный файл FastAPI приложения"""
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.model_manager import ModelManager
from app.core.cache import ModerationCache, NoOpModerationCache
from app.core.task_store import TaskStore
from app.core.db import init_db
from app.services.classification import ClassificationService
from app.loader import register_all_models, get_spam_model, get_spam_regex_model
from app.api.routes import router

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальные переменные для зависимостей
model_manager: ModelManager = None
classification_service: ClassificationService = None
task_store: TaskStore = None
_redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global model_manager, classification_service, task_store, _redis_client

    # Startup
    logger.info("Starting ToxicFilter service...")

    # Redis (опционально)
    if settings.redis_url:
        try:
            import redis
            _redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
            _redis_client.ping()
            moderation_cache = ModerationCache(_redis_client)
            task_store = TaskStore(_redis_client)
            logger.info("Redis connected, cache and task_store enabled")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory fallback")
            _redis_client = None
            moderation_cache = NoOpModerationCache()
            task_store = TaskStore(None)
    else:
        _redis_client = None
        moderation_cache = NoOpModerationCache()
        task_store = TaskStore(None)

    if settings.database_url:
        init_db()

    # Инициализация менеджера моделей
    model_manager = ModelManager()
    logger.info("Registering models...")
    register_all_models(model_manager)

    # Модели спама: regex pre-filter + TF-IDF (опционально)
    spam_regex_model = get_spam_regex_model()
    spam_model = get_spam_model()

    # Инициализация сервиса классификации (токсичность + спам, кэш)
    classification_service = ClassificationService(
        model_manager,
        moderation_cache=moderation_cache,
        spam_model=spam_model,
        spam_regex_model=spam_regex_model,
    )

    logger.info("Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down ToxicFilter service...")
    if _redis_client:
        try:
            _redis_client.close()
        except Exception:
            pass


# Создание FastAPI приложения
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(router, prefix=settings.api_prefix)

# Фронтенд: чат-демо
_frontend_dir = Path(__file__).resolve().parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="frontend")


@app.get("/chat", tags=["frontend"])
async def chat_page():
    """Редирект на страницу чата-демо."""
    return RedirectResponse(url="/static/index.html")


@app.get("/", tags=["root"])
async def root():
    """Корневой endpoint"""
    return {
        "service": "ToxicFilter",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )



