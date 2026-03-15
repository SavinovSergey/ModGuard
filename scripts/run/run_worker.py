#!/usr/bin/env python3
"""Воркер: потребляет очередь moderation.requests, запускает классификацию, пишет в Postgres и публикует в moderation.results."""
import logging
import os
import sys

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


def create_classification_service():
    from app.core.model_manager import ModelManager
    from app.services.classification import ClassificationService
    from app.loader import register_all_models, get_spam_model, get_spam_regex_model

    model_manager = ModelManager()
    register_all_models(model_manager)
    spam_model = get_spam_model()
    spam_regex_model = get_spam_regex_model()
    return ClassificationService(
        model_manager,
        spam_model=spam_model,
        spam_regex_model=spam_regex_model,
    )


def main():
    from app.core.config import settings
    from app.core.db import init_db, set_task_processing_pg, set_task_result_pg, set_task_failed_pg
    from app.core.queue import consume_requests, publish_task_result
    from scripts.run.run_listener_telegram import create_cache

    if not settings.rabbitmq_url:
        logger.error("RABBITMQ_URL is not set")
        sys.exit(1)
    if not settings.database_url:
        logger.error("DATABASE_URL is not set")
        sys.exit(1)

    init_db()

    cache = create_cache(redis_url=settings.redis_url)
    classification_service = create_classification_service()

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
            # Сохраняем результаты в кэш для последующих запросов (проверка кэша — на стороне API)
            n_written = 0
            for text, r in zip(texts, results):
                if text and (r.get("tox_model_used") or r.get("spam_model_used")):
                    cache.set_cached_result(text, r, tox_model_used=r.get("tox_model_used"))
                    n_written += 1
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
