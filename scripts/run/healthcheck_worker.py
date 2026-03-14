#!/usr/bin/env python3
"""
Healthcheck для backend (worker): создаёт задачу в Postgres, публикует в moderation.requests,
опрашивает Postgres по статусу (до 40 с). При успехе удаляет тестовую задачу из Postgres.
"""
import json
import logging
import os
import sys
import time
import uuid

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _root)
os.chdir(_root)

logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("pika.adapters").setLevel(logging.WARNING)

POLL_INTERVAL = 2
DEADLINE_SEC = 40


def main():
    import pika
    from app.core.config import settings
    from app.core.db import create_task_pg, get_task_pg, delete_task_pg

    if not settings.database_url:
        print("healthcheck_worker: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    task_id = str(uuid.uuid4())
    items_payload = [{"id": "0", "text": "healthcheck"}]
    body = {
        "task_id": task_id,
        "items": [{"text": "healthcheck", "ref": {}}],
        "source": "healthcheck",
    }

    create_task_pg(task_id, items_payload, source="healthcheck")

    url = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
    queue_requests = os.environ.get("RABBITMQ_QUEUE_REQUESTS", "moderation.requests")
    params = pika.URLParameters(url)
    params.socket_timeout = 5
    params.blocked_connection_timeout = 5

    try:
        conn = pika.BlockingConnection(params)
        ch = conn.channel()
        ch.queue_declare(queue=queue_requests, durable=True)
        ch.basic_publish(
            exchange="",
            routing_key=queue_requests,
            body=json.dumps(body, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        conn.close()
    except Exception as e:
        print(f"healthcheck_worker: publish failed: {e}", file=sys.stderr)
        sys.exit(1)

    deadline = time.time() + DEADLINE_SEC
    while time.time() < deadline:
        data = get_task_pg(task_id)
        if data is None:
            time.sleep(POLL_INTERVAL)
            continue
        status = data.get("status")
        if status == "completed":
            delete_task_pg(task_id)
            sys.exit(0)
        if status == "failed":
            print(f"healthcheck_worker: task {task_id} failed", file=sys.stderr)
            sys.exit(1)
        time.sleep(POLL_INTERVAL)

    print("healthcheck_worker: timeout waiting for task completion", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
