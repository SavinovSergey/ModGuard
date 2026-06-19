"""Unit-тесты async RabbitMQ publisher (aio-pika)."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import queue_async


@pytest.fixture(autouse=True)
def reset_queue_globals():
    queue_async._connection = None
    queue_async._channel = None
    yield
    queue_async._connection = None
    queue_async._channel = None


@pytest.mark.asyncio
async def test_publish_task_request_serializes_body():
    channel = MagicMock()
    channel.default_exchange.publish = AsyncMock()
    queue_async._channel = channel

    with patch.object(queue_async.settings, "rabbitmq_url", "amqp://test/"):
        with patch.object(queue_async.settings, "rabbitmq_queue_requests", "moderation.requests"):
            await queue_async.publish_task_request(
                "task-1",
                [{"text": "hello", "ref": {}}],
                source="web",
                preferred_model="regex",
            )

    channel.default_exchange.publish.assert_awaited_once()
    message = channel.default_exchange.publish.await_args.args[0]
    body = json.loads(message.body.decode("utf-8"))
    assert body["task_id"] == "task-1"
    assert body["items"][0]["text"] == "hello"
    assert body["preferred_model"] == "regex"


@pytest.mark.asyncio
async def test_publish_task_result_best_effort_no_channel():
    with patch.object(queue_async.settings, "rabbitmq_url", "amqp://test/"):
        # _channel is None — не падаем
        await queue_async.publish_task_result("t1", "web", [{"is_toxic": False}])


@pytest.mark.asyncio
async def test_publish_task_request_requires_channel():
    with patch.object(queue_async.settings, "rabbitmq_url", "amqp://test/"):
        with pytest.raises(RuntimeError, match="not initialized"):
            await queue_async.publish_task_request("t1", [])


@pytest.mark.asyncio
async def test_init_queue_publisher_declares_queues():
    conn = MagicMock()
    channel = MagicMock()
    channel.declare_queue = AsyncMock()
    conn.channel = AsyncMock(return_value=channel)

    with patch.object(queue_async.settings, "rabbitmq_url", "amqp://test/"):
        with patch.object(queue_async.settings, "rabbitmq_queue_requests", "req.q"):
            with patch.object(queue_async.settings, "rabbitmq_queue_results", "res.q"):
                with patch("app.core.queue_async.aio_pika.connect_robust", new=AsyncMock(return_value=conn)):
                    ok = await queue_async.init_queue_publisher()

    assert ok is True
    assert channel.declare_queue.await_count == 2
