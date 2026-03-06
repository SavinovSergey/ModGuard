#!/usr/bin/env python3
"""
Actions Telegram: потребляет очередь moderation.results,
для сообщений с source=telegram выполняет действия по политике (удаление при токсичности/спаме).
"""
import logging
import os
import sys
import requests


_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _root)
os.chdir(_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def delete_message(chat_id: int, message_id: int, token: str) -> bool:
    url = TELEGRAM_API.format(token=token, method="deleteMessage")
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "message_id": message_id},
            timeout=10,
        )
        if not r.ok:
            logger.warning("deleteMessage failed: %s %s", r.status_code, r.text)
            return False
        return True
    except Exception as e:
        logger.warning("deleteMessage error: %s", e)
        return False


def main():
    from app.core.config import settings
    from app.core.queue import consume_results
    
    token = settings.telegram_bot_token
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set")
        sys.exit(1)
    if not settings.rabbitmq_url:
        logger.error("RABBITMQ_URL is not set")
        sys.exit(1)


    def on_result(body: dict):
        source = body.get("source")
        if source != "telegram":
            return
        results = body.get("results") or []
        for item in results:
            ref = item.get("ref")
            if not ref or "chat_id" not in ref or "message_id" not in ref:
                continue
            chat_id = ref["chat_id"]
            message_id = ref["message_id"]
            is_toxic = item.get("is_toxic", False)
            is_spam = item.get("is_spam", False)
            # Политика: токсичное или спам — удаляем сообщение
            if is_toxic or is_spam:
                if delete_message(chat_id, message_id):
                    logger.info("Deleted message %s in chat %s (toxic=%s spam=%s)", message_id, chat_id, is_toxic, is_spam)
                else:
                    logger.warning("Could not delete message %s in chat %s", message_id, chat_id)

    logger.info("Telegram actions started, consuming results...")
    consume_results(on_result)


if __name__ == "__main__":
    main()
