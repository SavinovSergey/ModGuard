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
from app.core.cache import ModerationCache, NoOpModerationCache
from app.core.db import init_db
from app.api.routes import router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("pika.adapters").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

task_store: TaskStore = None
_redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом (API: только очередь + task store)."""
    global task_store, _redis_client

    logger.info("Starting ToxicFilter API (queue + task_id only)...")

    if settings.redis_url:
        try:
            import redis
            _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            _redis_client.ping()
            task_store = TaskStore(_redis_client)
            app.state.moderation_cache = ModerationCache(_redis_client)
            logger.info("Redis connected, task_store and moderation_cache enabled")
        except Exception as e:
            logger.warning("Redis connection failed: %s, using in-memory fallback", e)
            _redis_client = None
            task_store = TaskStore(None)
            app.state.moderation_cache = NoOpModerationCache()
    else:
        _redis_client = None
        task_store = TaskStore(None)
        app.state.moderation_cache = NoOpModerationCache()

    if settings.database_url:
        init_db()

    logger.info("API started (submit to queue, GET /tasks/{task_id} for result)")
    yield

    logger.info("Shutting down ToxicFilter API...")
    if _redis_client:
        try:
            _redis_client.close()
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
        "service": "ToxicFilter",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.reload)
