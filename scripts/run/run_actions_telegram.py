#!/usr/bin/env python3
"""
Actions Telegram: потребляет очередь moderation.results,
для сообщений с source=telegram выполняет действия по политике (удаление при токсичности/спаме).
"""
import logging
import os
import sys
import time
import requests


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


def delete_message(chat_id: int, message_id: int, token: str) -> bool:
    url = TELEGRAM_API.format(token=token, method="deleteMessage")
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "message_id": message_id},
            timeout=10,
        )
        if not r.ok:
            err = r.text
            logger.warning(
                "deleteMessage failed: %s %s. Убедитесь, что бот добавлен в группу как администратор с правом «Удаление сообщений».",
                r.status_code,
                err,
            )
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
        task_id = body.get("task_id", "?")
        source = body.get("source")
        # Логируем каждое входящее сообщение из очереди результатов (диагностика)
        logger.info("Received result: task_id=%s source=%s", task_id, source)
        if source != "telegram":
            return
        results = body.get("results") or []
        logger.info("Result task_id=%s source=telegram items=%d", task_id, len(results))
        for item in results:
            ref = item.get("ref")
            if not ref or "chat_id" not in ref or "message_id" not in ref:
                logger.warning("Skip item: ref missing or invalid (ref=%s)", ref)
                continue
            chat_id = ref["chat_id"]
            message_id = ref["message_id"]
            is_toxic = item.get("is_toxic", False)
            is_spam = item.get("is_spam", False)
            if is_toxic or is_spam:
                logger.info("Action: delete msg_id=%s chat_id=%s (is_toxic=%s is_spam=%s)", message_id, chat_id, is_toxic, is_spam)
                if delete_message(chat_id, message_id, token):
                    logger.info("Deleted message %s in chat %s", message_id, chat_id)
                else:
                    logger.warning("Could not delete message %s in chat %s", message_id, chat_id)
            else:
                logger.info("Skip: not toxic/spam (is_toxic=%s is_spam=%s) msg_id=%s — модель не пометила как нарушение", is_toxic, is_spam, message_id)

    logger.info("Telegram actions started, connecting to RabbitMQ...")
    while True:
        try:
            consume_results(on_result)
        except Exception as e:
            errname = type(e).__name__
            if errname == "AMQPConnectionError" or "Connection refused" in str(e).lower():
                logger.warning("RabbitMQ connection failed (retry in 5s): %s", e)
            else:
                logger.exception("Consume error: %s", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
