"""Sync RabbitMQ (pika): consumer и publisher для worker/scripts. Один channel на процесс."""
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_pika_conn = None
_pika_channel = None


def _get_publish_channel():
    """Lazy singleton channel; queue_declare один раз."""
    global _pika_conn, _pika_channel
    if not settings.rabbitmq_url:
        return None
    import pika

    if _pika_channel is not None and _pika_conn is not None and _pika_conn.is_open:
        return _pika_channel
    params = pika.URLParameters(settings.rabbitmq_url)
    _pika_conn = pika.BlockingConnection(params)
    _pika_channel = _pika_conn.channel()
    _pika_channel.queue_declare(queue=settings.rabbitmq_queue_requests, durable=True)
    _pika_channel.queue_declare(queue=settings.rabbitmq_queue_results, durable=True)
    return _pika_channel


def publish_task_request(
    task_id: str,
    items: List[Dict[str, Any]],
    source: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> None:
    """Sync publish (listener/scripts)."""
    if not settings.rabbitmq_url:
        return
    import pika

    try:
        ch = _get_publish_channel()
        if ch is None:
            return
        body = {
            "task_id": task_id,
            "items": items,
            "source": source,
            "preferred_model": preferred_model,
        }
        ch.basic_publish(
            exchange="",
            routing_key=settings.rabbitmq_queue_requests,
            body=json.dumps(body, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=2),
        )
    except Exception as e:
        logger.warning("publish_task_request failed: %s", e)
        raise


def publish_task_result(
    task_id: str,
    source: Optional[str],
    results: List[Dict[str, Any]],
) -> None:
    """Sync publish результата (worker)."""
    if not settings.rabbitmq_url:
        return
    import pika

    try:
        ch = _get_publish_channel()
        if ch is None:
            return
        body = {
            "task_id": task_id,
            "source": source,
            "results": results,
        }
        ch.basic_publish(
            exchange="",
            routing_key=settings.rabbitmq_queue_results,
            body=json.dumps(body, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=2),
        )
    except Exception as e:
        logger.warning("publish_task_result failed: %s", e)


def consume_requests(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Бесконечно потребляет из moderation.requests и вызывает callback(body_dict)."""
    import pika

    params = pika.URLParameters(settings.rabbitmq_url)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=settings.rabbitmq_queue_requests, durable=True)
    ch.basic_qos(prefetch_count=1)

    def _on_message(channel, method, properties, body):
        try:
            data = json.loads(body)
            callback(data)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.exception("Worker callback error: %s", e)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    ch.basic_consume(
        queue=settings.rabbitmq_queue_requests,
        on_message_callback=_on_message,
    )
    logger.info("Worker consuming from %s", settings.rabbitmq_queue_requests)
    ch.start_consuming()


def consume_results(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Бесконечно потребляет из moderation.results и вызывает callback(body_dict)."""
    import pika

    params = pika.URLParameters(settings.rabbitmq_url)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=settings.rabbitmq_queue_results, durable=True)
    ch.basic_qos(prefetch_count=1)

    def _on_message(channel, method, properties, body):
        try:
            data = json.loads(body)
            callback(data)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.exception("Actions callback error: %s", e)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    ch.basic_consume(
        queue=settings.rabbitmq_queue_results,
        on_message_callback=_on_message,
    )
    logger.info("Consuming from %s", settings.rabbitmq_queue_results)
    ch.start_consuming()
