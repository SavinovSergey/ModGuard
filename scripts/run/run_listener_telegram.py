#!/usr/bin/env python3
"""
Listener Telegram: получает апдейты от Telegram (long polling),
проверяет кэш по тексту; при промахе публикует задачу в RabbitMQ,
при попадании в кэш пишет результат в Postgres и публикует в очередь результатов.
"""
import logging
import os
import sys
import time

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _root)
os.chdir(_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("pika.adapters").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def create_cache(redis_url: str | None):
    """Возвращает ModerationCache или NoOpModerationCache."""
    from app.core.cache import ModerationCache, NoOpModerationCache
    if not redis_url:
        return NoOpModerationCache()
    try:
        import redis
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return ModerationCache(client)
    except Exception as e:
        logger.warning("Redis failed: %s, cache disabled", e)
        return NoOpModerationCache()


def process_updates(updates: list, offset: int, cache, token: str, database_url: str | None) -> int:
    """Обрабатывает пакет апдейтов: кэш, создание задачи, публикация в очередь. Возвращает новый offset."""
    import uuid
    from app.core.db import create_task_pg, set_task_result_pg
    from app.core.queue import publish_task_request, publish_task_result

    source = "telegram"
    for upd in updates:
        offset = upd["update_id"] + 1
        msg = upd.get("message") or upd.get("edited_message")
        if not msg or "text" not in msg:
            continue
        text = msg.get("text", "").strip()
        if not text:
            continue
        chat_id = msg.get("chat", {}).get("id")
        message_id = msg.get("message_id")
        from_user = msg.get("from") or {}
        user_id = from_user.get("id")  # Telegram user id (int)
        ref = {"chat_id": chat_id, "message_id": message_id}
        task_id = str(uuid.uuid4())
        items = [{"id": str(message_id), "text": text, "ref": ref}]

        cached = cache.get_cached_result(text) if cache else None
        if cached is not None:
            if database_url:
                create_task_pg(task_id, items, source=source, user_id=user_id)
                set_task_result_pg(task_id, [cached], status="completed", from_cache=True)
            # Передаём ref в результат, чтобы actions-telegram мог удалить сообщение при необходимости
            result_with_ref = {**cached, "ref": ref}
            publish_task_result(task_id, source, [result_with_ref])
            logger.debug("Task %s from cache", task_id)
        else:
            if database_url:
                create_task_pg(task_id, items, source=source, user_id=user_id)
            publish_task_request(
                task_id,
                [{"text": text, "ref": ref}],
                source=source,
            )
            logger.info("Task %s queued for backend", task_id)
    return offset


def run_listener_loop(token: str, cache, database_url: str | None) -> None:
    """Бесконечный цикл: getUpdates → process_updates."""
    import requests

    url_get = TELEGRAM_API.format(token=token, method="getUpdates")
    offset = 0
    last_alive_log = 0.0
    logger.info("Telegram listener started, long polling...")

    while True:
        try:
            r = requests.get(
                url_get,
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                logger.warning("Telegram API not ok: %s", data)
                time.sleep(5)
                continue
            updates = data.get("result") or []
            offset = process_updates(updates, offset, cache, token, database_url)
            # Раз в минуту пишем в лог, что listener жив (если нет сообщений — в логах иначе пусто)
            if not updates and time.time() - last_alive_log >= 60:
                logger.info("Listener alive, waiting for messages (offset=%s). If no messages in groups: disable Privacy Mode in BotFather -> /setprivacy -> your bot -> Disable.", offset)
                last_alive_log = time.time()
        except requests.RequestException as e:
            logger.warning("Telegram request error: %s", e)
            time.sleep(5)
        except Exception as e:
            logger.exception("Listener error: %s", e)
            time.sleep(5)


def main():
    from app.core.config import settings
    from app.core.db import init_db

    token = settings.telegram_bot_token
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set")
        sys.exit(1)
    if not settings.rabbitmq_url:
        logger.error("RABBITMQ_URL is not set")
        sys.exit(1)

    logger.info("Starting listener (token set, rabbitmq required). In groups, disable bot Privacy Mode in BotFather so the bot receives all messages.")
    cache = create_cache(redis_url=settings.redis_url)
    if settings.database_url and not init_db():
        logger.error(
            "Postgres init failed. Fix DATABASE_URL or run: python scripts/run/init_postgres.py"
        )
        sys.exit(1)

    run_listener_loop(token, cache, settings.database_url)


if __name__ == "__main__":
    main()
