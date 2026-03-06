"""Публикация и потребление очередей RabbitMQ (moderation.requests / moderation.results)."""
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def publish_task_request(
    task_id: str,
    items: List[Dict[str, Any]],
    source: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> None:
    """Публикует задачу в очередь moderation.requests."""
    if not settings.rabbitmq_url:
        return
    try:
        import pika
        body = {
            "task_id": task_id,
            "items": items,
            "source": source,
            "preferred_model": preferred_model,
        }
        params = pika.URLParameters(settings.rabbitmq_url)
        conn = pika.BlockingConnection(params)
        ch = conn.channel()
        ch.queue_declare(queue=settings.rabbitmq_queue_requests, durable=True)
        ch.basic_publish(
            exchange="",
            routing_key=settings.rabbitmq_queue_requests,
            body=json.dumps(body, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        conn.close()
    except Exception as e:
        logger.warning("publish_task_request failed: %s", e)
        raise


def publish_task_result(
    task_id: str,
    source: Optional[str],
    results: List[Dict[str, Any]],
) -> None:
    """Публикует результат в очередь moderation.results (для Action-сервисов)."""
    if not settings.rabbitmq_url:
        return
    try:
        import pika
        body = {
            "task_id": task_id,
            "source": source,
            "results": results,
        }
        params = pika.URLParameters(settings.rabbitmq_url)
        conn = pika.BlockingConnection(params)
        ch = conn.channel()
        ch.queue_declare(queue=settings.rabbitmq_queue_results, durable=True)
        ch.basic_publish(
            exchange="",
            routing_key=settings.rabbitmq_queue_results,
            body=json.dumps(body, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        conn.close()
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
