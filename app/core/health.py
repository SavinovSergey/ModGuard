"""Проверка доступности зависимостей API (Postgres, Redis, RabbitMQ)."""
import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_SEC = 2.0


async def _check_postgres() -> Dict[str, Any]:
    if not settings.database_url:
        return {"status": "skipped", "reason": "not configured"}
    try:
        from app.core.db import get_pool

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _check_redis(redis_client: Optional[Any]) -> Dict[str, Any]:
    if not settings.redis_url:
        return {"status": "skipped", "reason": "not configured"}
    if redis_client is None:
        return {"status": "error", "error": "client unavailable"}
    try:
        await redis_client.ping()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _check_rabbitmq() -> Dict[str, Any]:
    if not settings.rabbitmq_url:
        return {"status": "skipped", "reason": "not configured"}
    try:
        from app.core import queue_async

        conn = queue_async._connection
        ch = queue_async._channel
        if conn is None or ch is None:
            return {"status": "error", "error": "not connected"}
        if conn.is_closed:
            return {"status": "error", "error": "connection closed"}
        if ch.is_closed:
            return {"status": "error", "error": "channel closed"}
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _aggregate_status(checks: Dict[str, Dict[str, Any]]) -> str:
    """healthy — всё настроенное доступно; degraded — Redis недоступен; unhealthy — PG/MQ."""
    if settings.database_url and checks.get("postgres", {}).get("status") == "error":
        return "unhealthy"
    if settings.rabbitmq_url and checks.get("rabbitmq", {}).get("status") == "error":
        return "unhealthy"
    if settings.redis_url and checks.get("redis", {}).get("status") == "error":
        return "degraded"
    return "healthy"


async def check_dependencies(redis_client: Optional[Any] = None) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Параллельный ping зависимостей. Возвращает (overall_status, checks)."""
    try:
        async with asyncio.timeout(CHECK_TIMEOUT_SEC):
            pg, redis, mq = await asyncio.gather(
                _check_postgres(),
                _check_redis(redis_client),
                _check_rabbitmq(),
            )
    except asyncio.TimeoutError:
        logger.warning("Health check timed out after %ss", CHECK_TIMEOUT_SEC)
        return "unhealthy", {
            "postgres": {"status": "error", "error": "health check timeout"},
            "redis": {"status": "error", "error": "health check timeout"},
            "rabbitmq": {"status": "error", "error": "health check timeout"},
        }

    checks = {"postgres": pg, "redis": redis, "rabbitmq": mq}
    return _aggregate_status(checks), checks
