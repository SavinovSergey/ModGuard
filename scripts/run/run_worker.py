#!/usr/bin/env python3
"""Воркер: потребляет очередь moderation.requests, запускает классификацию, пишет в Postgres и публикует в moderation.results."""
import logging
import os
import sys
from typing import Union

# рабочий каталог — корень проекта (где models/)
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


from app.core.cache import ModerationCache, NoOpModerationCache
from scripts.run.run_listener_telegram import create_cache


def create_classification_service(cache : Union[ModerationCache, NoOpModerationCache]):
    from app.core.model_manager import ModelManager
    from app.services.classification import ClassificationService
    from app.loader import register_all_models, get_spam_model, get_spam_regex_model

    model_manager = ModelManager()
    register_all_models(model_manager)
    spam_model = get_spam_model()
    spam_regex_model = get_spam_regex_model()
    classification_service = ClassificationService(
        model_manager,
        moderation_cache=cache,
        spam_model=spam_model,
        spam_regex_model=spam_regex_model,
    )
    return classification_service


def main():
    from app.core.config import settings
    from app.core.db import init_db, set_task_processing_pg, set_task_result_pg, set_task_failed_pg
    from app.core.queue import consume_requests, publish_task_result

    if not settings.rabbitmq_url:
        logger.error("RABBITMQ_URL is not set")
        sys.exit(1)
    if not settings.database_url:
        logger.error("DATABASE_URL is not set")
        sys.exit(1)

    init_db()

    # Кэш и модели
    cache = create_cache(redis_url=settings.redis_url)
    classification_service = create_classification_service(cache=cache)

    def on_message(body: dict):
        task_id = body.get("task_id")
        items = body.get("items") or []
        source = body.get("source")
        preferred_model = body.get("preferred_model")
        texts = [item.get("text", "") for item in items]
        logger.info("Processing task %s, %d items", task_id, len(texts))
        try:
            set_task_processing_pg(task_id)
            results = classification_service.classify_batch(texts, preferred_model=preferred_model)
            set_task_result_pg(task_id, results, status="completed")
            # Для Action-сервисов передаём ref (chat_id, message_id и т.д.) в каждом элементе
            out_results = []
            tox_used = [r.get("tox_model_used") for r in results]
            spam_used = [r.get("spam_model_used") for r in results]
            for i, r in enumerate(results):
                ref = items[i].get("ref") if i < len(items) else None
                out_results.append({"ref": ref, **r})
            publish_task_result(task_id, source, out_results)
            logger.info(
                "Task %s completed, tox_model_used=%s spam_model_used=%s",
                task_id,
                tox_used,
                spam_used,
            )
        except Exception as e:
            logger.exception("Task %s failed: %s", task_id, e)
            set_task_failed_pg(task_id, str(e))

    consume_requests(on_message)


if __name__ == "__main__":
    main()
