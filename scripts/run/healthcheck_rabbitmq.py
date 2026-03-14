#!/usr/bin/env python3
"""
Healthcheck: проверка подключения к RabbitMQ (для listener-telegram и actions-telegram).
Выход 0 — подключение успешно, иначе 1.
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _root)
os.chdir(_root)


def main():
    import pika
    url = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
    params = pika.URLParameters(url)
    params.socket_timeout = 5
    params.blocked_connection_timeout = 5
    try:
        conn = pika.BlockingConnection(params)
        conn.close()
    except Exception as e:
        print(f"healthcheck_rabbitmq: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
