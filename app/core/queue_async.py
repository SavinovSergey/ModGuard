"""Async RabbitMQ (aio-pika): publisher и consumer, один channel на процесс."""
import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional

import aio_pika
from aio_pika import DeliveryMode, Message

from app.core.config import settings

logger = logging.getLogger(__name__)

_connection: Optional[aio_pika.RobustConnection] = None
_channel: Optional[aio_pika.Channel] = None

MessageHandler = Callable[[Dict[str, Any]], Awaitable[None]]


async def init_queue_publisher() -> bool:
    """Подключение и declare очередей один раз при старте."""
    global _connection, _channel
    if not settings.rabbitmq_url:
        return False
    try:
        _connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        _channel = await _connection.channel()
        await _channel.declare_queue(settings.rabbitmq_queue_requests, durable=True)
        await _channel.declare_queue(settings.rabbitmq_queue_results, durable=True)
        logger.info("RabbitMQ async client ready (shared channel)")
        return True
    except Exception as e:
        logger.warning("RabbitMQ async init failed: %s", e)
        _connection = None
        _channel = None
        return False


async def close_queue_publisher() -> None:
    global _connection, _channel
    if _channel is not None:
        try:
            await _channel.close()
        except Exception:
            pass
        _channel = None
    if _connection is not None:
        try:
            await _connection.close()
        except Exception:
            pass
        _connection = None


def _require_channel() -> aio_pika.Channel:
    if _channel is None:
        raise RuntimeError("RabbitMQ async client not initialized")
    return _channel


async def publish_task_request(
    task_id: str,
    items: List[Dict[str, Any]],
    source: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> None:
    """Публикует задачу в moderation.requests."""
    if not settings.rabbitmq_url:
        return
    channel = _require_channel()
    body = {
        "task_id": task_id,
        "items": items,
        "source": source,
        "preferred_model": preferred_model,
    }
    try:
        await channel.default_exchange.publish(
            Message(
                body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=settings.rabbitmq_queue_requests,
        )
    except Exception as e:
        logger.warning("publish_task_request failed: %s", e)
        raise


async def publish_task_result(
    task_id: str,
    source: Optional[str],
    results: List[Dict[str, Any]],
) -> None:
    """Публикует результат в moderation.results (best-effort)."""
    if not settings.rabbitmq_url or _channel is None:
        return
    body = {
        "task_id": task_id,
        "source": source,
        "results": results,
    }
    try:
        await _channel.default_exchange.publish(
            Message(
                body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=settings.rabbitmq_queue_results,
        )
    except Exception as e:
        logger.warning("publish_task_result failed: %s", e)


async def consume_requests(
    handler: MessageHandler,
    prefetch_count: int = 1,
) -> None:
    """Потребляет moderation.requests, обрабатывая до prefetch_count сообщений конкурентно.

    Конкурентность нужна, чтобы I/O одного батча (Postgres/Redis/RabbitMQ) перекрывался
    с вычислением другого, а не суммировался последовательно. Число одновременных
    обработчиков ограничено prefetch_count (он же qos), чтобы не выгребать всю очередь.
    """
    channel = _require_channel()
    await channel.set_qos(prefetch_count=prefetch_count)
    queue = await channel.declare_queue(settings.rabbitmq_queue_requests, durable=True)
    logger.info(
        "Worker consuming from %s (prefetch=%s, concurrent)",
        settings.rabbitmq_queue_requests,
        prefetch_count,
    )

    in_flight: set[asyncio.Task] = set()

    async def _process_one(message: aio_pika.IncomingMessage) -> None:
        async with message.process(requeue=True):
            data = json.loads(message.body.decode("utf-8"))
            await handler(data)

    try:
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                task = asyncio.create_task(_process_one(message))
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)
                if len(in_flight) >= prefetch_count:
                    await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
    finally:
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)
