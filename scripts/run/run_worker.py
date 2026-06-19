#!/usr/bin/env python3
"""Async worker: moderation.requests → classify → Postgres/Redis → moderation.results."""
import asyncio
import logging
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _root)
os.chdir(_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("aio_pika").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def create_classification_service():
    from app.services.classification import ClassificationService

    return ClassificationService()


async def process_task(
    body: dict,
    cache,
    classification_service,
) -> None:
    from app.core import chain_timing
    from app.core.db import set_task_failed_pg, set_task_processing_pg, set_task_result_pg
    from app.core.queue_async import publish_task_result

    task_id = body.get("task_id")
    items = body.get("items") or []
    source = body.get("source")
    preferred_model = body.get("preferred_model")
    texts = [item.get("text", "") for item in items]
    n_items = len(texts)
    logger.info("Processing task %s, %d items", task_id, n_items)

    chain_timing.mark("worker", "mq_consume", "start", task_id=task_id, n_items=n_items)
    try:
        with chain_timing.stage("worker", "task", task_id=task_id, n_items=n_items):
            await set_task_processing_pg(task_id)
            loop = asyncio.get_running_loop()
            with chain_timing.stage("worker", "classify", task_id=task_id, n_items=n_items):
                results = await loop.run_in_executor(
                    None,
                    lambda: classification_service.classify_batch(texts, preferred_model=preferred_model),
                )
            with chain_timing.stage("worker", "pg_result", task_id=task_id, n_items=n_items):
                await set_task_result_pg(task_id, results, status="completed")

            cache_pairs = [
                (text, r)
                for text, r in zip(texts, results)
                if text and (r.get("tox_model_used") or r.get("spam_model_used"))
            ]
            with chain_timing.stage("worker", "redis_cache", task_id=task_id, n_items=len(cache_pairs)):
                n_written = await cache.aset_cached_results_batch(cache_pairs)

            out_results = []
            for i, r in enumerate(results):
                ref = items[i].get("ref") if i < len(items) else None
                out_results.append({"ref": ref, **r})
            with chain_timing.stage("worker", "mq_publish_result", task_id=task_id, n_items=n_items):
                await publish_task_result(task_id, source, out_results)
        logger.info("Task %s completed, cache_written=%d", task_id, n_written)
    except Exception as e:
        logger.exception("Task %s failed: %s", task_id, e)
        await set_task_failed_pg(task_id, str(e))


async def async_main() -> None:
    from app.core.config import settings
    from app.core.cache import create_async_moderation_cache
    from app.core.db import close_pool, init_db, init_pool
    from app.core.queue_async import close_queue_publisher, consume_requests, init_queue_publisher

    if not settings.rabbitmq_url:
        logger.error("RABBITMQ_URL is not set")
        sys.exit(1)
    if not settings.database_url:
        logger.error("DATABASE_URL is not set")
        sys.exit(1)

    if not init_db():
        logger.error(
            "Postgres init failed. Fix DATABASE_URL or run: python scripts/run/init_postgres.py"
        )
        sys.exit(1)
    if not await init_pool():
        logger.error("Postgres async pool init failed")
        sys.exit(1)
    if settings.moderation_pipeline != "both":
        logger.warning(
            "MODERATION_PIPELINE=%s: disabled branch returns safe stubs (benchmark mode)",
            settings.moderation_pipeline,
        )
    if not await init_queue_publisher():
        logger.error("RabbitMQ init failed. Check RABBITMQ_URL")
        sys.exit(1)

    cache = await create_async_moderation_cache(settings.redis_url)
    classification_service = create_classification_service()
    logger.info("ClassificationService: both-mode uses ProcessPool (toxicity + spam)")

    async def on_message(body: dict) -> None:
        await process_task(body, cache, classification_service)

    try:
        await consume_requests(on_message, prefetch_count=12)
    finally:
        classification_service.shutdown()
        await close_queue_publisher()
        await close_pool()
        redis_client = getattr(cache, "_redis", None)
        if redis_client is not None:
            try:
                await redis_client.aclose()
            except Exception:
                pass


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
