"""Главный файл FastAPI приложения (только очередь и выдача по task_id; классификация в backend)."""
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.task_store import TaskStore
from app.core.cache import NoOpModerationCache, create_async_moderation_cache
from app.core.db import init_db, init_pool, close_pool
from app.core.queue_async import init_queue_publisher, close_queue_publisher
from app.api.routes import router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("pika.adapters").setLevel(logging.WARNING)
logging.getLogger("aio_pika").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

task_store: TaskStore = None
_redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом (API: только очередь + task store)."""
    global task_store, _redis_client

    logger.info("Starting ModGuard API (queue + task_id only)...")

    if settings.redis_url:
        app.state.moderation_cache = await create_async_moderation_cache(settings.redis_url)
        _redis_client = app.state.moderation_cache._redis
        task_store = TaskStore(_redis_client) if _redis_client is not None else TaskStore(None)
        if _redis_client is not None:
            logger.info("Redis async connected, task_store and moderation_cache enabled")
        else:
            logger.warning("Redis unavailable, using in-memory fallback")
    else:
        _redis_client = None
        task_store = TaskStore(None)
        app.state.moderation_cache = NoOpModerationCache()

    if settings.database_url:
        if not init_db():
            raise RuntimeError(
                "Не удалось инициализировать Postgres (БД и/или таблицы). "
                "Проверьте DATABASE_URL, что сервер запущен, и права пользователя "
                "(для автосоздания БД — CREATEDB). Либо выполните: python scripts/run/init_postgres.py"
            )
        if not await init_pool():
            raise RuntimeError("Не удалось создать пул соединений Postgres (asyncpg).")

    if settings.rabbitmq_url:
        if not await init_queue_publisher():
            raise RuntimeError(
                "Не удалось подключиться к RabbitMQ. Проверьте RABBITMQ_URL."
            )

    logger.info("API started (submit to queue, GET /tasks/{task_id} for result)")
    yield

    logger.info("Shutting down ModGuard API...")
    await close_queue_publisher()
    await close_pool()
    if _redis_client:
        try:
            await _redis_client.aclose()
        except Exception:
            pass


app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api_prefix)

_frontend_dir = Path(__file__).resolve().parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="frontend")


@app.get("/chat", tags=["frontend"])
async def chat_page():
    return RedirectResponse(url="/static/index.html")


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "ModGuard",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.reload)
