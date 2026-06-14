"""Postgres: таблицы tasks (батч) и task_items (по одному сообщению), слой доступа."""
import asyncio
import json
import asyncpg
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from app.core.config import settings

logger = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id VARCHAR(64) PRIMARY KEY,
    status VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);

CREATE TABLE IF NOT EXISTS task_items (
    task_id VARCHAR(64) NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    item_index INT NOT NULL,
    user_id BIGINT,
    source VARCHAR(64),
    text TEXT,
    from_cache BOOLEAN DEFAULT FALSE,
    tox_model_used VARCHAR(64),
    spam_model_used VARCHAR(64),
    is_toxic BOOLEAN,
    toxicity_score REAL,
    toxicity_types JSONB,
    is_spam BOOLEAN,
    spam_score REAL,
    error TEXT,
    PRIMARY KEY (task_id, item_index)
);
CREATE INDEX IF NOT EXISTS idx_task_items_user_id ON task_items(user_id);
CREATE INDEX IF NOT EXISTS idx_task_items_source ON task_items(source);
"""

_pool: asyncpg.Pool | None = None

async def init_pool() -> bool:
    global _pool
    if not settings.database_url:
        return False
    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=2,
        max_size=10,
        command_timeout=30.0,
    )
    return True

async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized")
    return _pool


_db_loop: asyncio.AbstractEventLoop | None = None


def run_db(coro):
    """Выполнить async-корутину БД из синхронного кода (worker, scripts)."""
    global _db_loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if _db_loop is None or _db_loop.is_closed():
            _db_loop = asyncio.new_event_loop()
        return _db_loop.run_until_complete(coro)
    raise RuntimeError("run_db() cannot be called from a running event loop")

async def init_db_tables() -> bool:
    if not settings.database_url:
        return False
    try:
        ensure_postgres_database_exists(settings.database_url)
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(CREATE_TABLES_SQL)
        return True
    except Exception as e:
        logger.warning("init_db_tables failed: %s", e)
        return False

def _get_connection():
    import psycopg2
    return psycopg2.connect(settings.database_url)


def ensure_postgres_database_exists(database_url: str) -> None:
    """
    Если в DATABASE_URL указана БД, которой ещё нет на сервере, создаёт её.
    Подключается к служебной БД postgres (те же host/user/password/query из URL).
    Нужны права CREATEDB или суперпользователь. Для не-postgres схем URL — no-op.
    """
    parsed = urlparse(database_url)
    if parsed.scheme not in ("postgresql", "postgres"):
        return
    path = (parsed.path or "").strip("/")
    if not path:
        return
    db_name = path.split("/")[0]
    if not db_name or db_name in ("postgres", "template0", "template1"):
        return

    import psycopg2
    from psycopg2 import sql

    maintenance = urlunparse(
        (parsed.scheme, parsed.netloc, "/postgres", "", parsed.query, parsed.fragment)
    )
    conn = psycopg2.connect(maintenance)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone():
                return
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            logger.info("Postgres: created database %s", db_name)
    finally:
        conn.close()


@contextmanager
def get_db_connection():
    """Контекстный менеджер соединения с Postgres."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> bool:
    """
    Создаёт БД (если нужно) и таблицы tasks/task_items, если их ещё нет.
    Возвращает True при успехе. Данные не удаляются.
    """
    if not settings.database_url:
        return False
    try:
        ensure_postgres_database_exists(settings.database_url)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLES_SQL)
        logger.info("Postgres: tasks and task_items ready")
        return True
    except Exception as e:
        logger.warning("Postgres init_db failed: %s", e)
        return False


async def create_task_pg(
    task_id: str,
    items: List[Dict[str, Any]],
    source: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    """Создаёт запись батча в tasks и по одной строке в task_items на каждый item (item_index 0-based)."""
    if not settings.database_url:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO tasks (task_id, status)
                    VALUES ($1, 'queued')
                    ON CONFLICT (task_id) DO UPDATE SET status = 'queued'
                    """,
                    task_id,
                )
                await _bulk_insert_task_items(conn, task_id, items, source, user_id)
    except Exception as e:
        logger.warning("create_task_pg failed: %s", e)
        raise


async def set_task_processing_pg(task_id: str) -> None:
    if not settings.database_url:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET status = 'processing' WHERE task_id = $1", 
                task_id,
            )
    except Exception as e:
        logger.warning("set_task_processing_pg failed: %s", e)


def _result_to_row(r: Dict[str, Any], from_cache: bool = False) -> tuple:
    """Из результата классификации формирует кортеж для UPDATE task_items."""
    tox_types = r.get("toxicity_types")
    if isinstance(tox_types, dict):
        tox_types = json.dumps(tox_types, ensure_ascii=False)
    return (
        from_cache,
        r.get("tox_model_used"),
        r.get("spam_model_used"),
        bool(r.get("is_toxic", False)),
        float(r.get("toxicity_score", 0.0)),
        tox_types,
        bool(r.get("is_spam", False)),
        float(r.get("spam_score", 0.0)),
        r.get("error"),
    )


def _results_to_update_arrays(
    results: List[Dict[str, Any]],
    from_cache: bool,
) -> tuple[
    List[bool],
    List[Optional[str]],
    List[Optional[str]],
    List[bool],
    List[float],
    List[Optional[str]],
    List[bool],
    List[float],
    List[Optional[str]],
    List[int],
]:
    """Колоночные массивы для bulk UPDATE task_items через unnest."""
    from_cache_a: List[bool] = []
    tox_model_a: List[Optional[str]] = []
    spam_model_a: List[Optional[str]] = []
    is_toxic_a: List[bool] = []
    tox_score_a: List[float] = []
    tox_types_a: List[Optional[str]] = []
    is_spam_a: List[bool] = []
    spam_score_a: List[float] = []
    error_a: List[Optional[str]] = []
    item_index_a: List[int] = []

    for i, result in enumerate(results):
        row = _result_to_row(result, from_cache)
        item_index_a.append(i)
        from_cache_a.append(row[0])
        tox_model_a.append(row[1])
        spam_model_a.append(row[2])
        is_toxic_a.append(row[3])
        tox_score_a.append(row[4])
        tox_types_a.append(row[5])
        is_spam_a.append(row[6])
        spam_score_a.append(row[7])
        error_a.append(row[8])

    return (
        from_cache_a,
        tox_model_a,
        spam_model_a,
        is_toxic_a,
        tox_score_a,
        tox_types_a,
        is_spam_a,
        spam_score_a,
        error_a,
        item_index_a,
    )


async def _bulk_insert_task_items(
    conn: asyncpg.Connection,
    task_id: str,
    items: List[Dict[str, Any]],
    source: Optional[str],
    user_id: Optional[int],
) -> None:
    if not items:
        return
    item_indices = list(range(len(items)))
    texts = [item.get("text", "") for item in items]
    await conn.execute(
        """
        INSERT INTO task_items (task_id, item_index, user_id, source, text, from_cache)
        SELECT $1, u.item_index, $2, $3, u.text, FALSE
        FROM unnest($4::int[], $5::text[]) AS u(item_index, text)
        ON CONFLICT (task_id, item_index) DO UPDATE SET
            user_id = EXCLUDED.user_id,
            source = EXCLUDED.source,
            text = EXCLUDED.text,
            from_cache = FALSE
        """,
        task_id,
        user_id,
        source,
        item_indices,
        texts,
    )


async def _bulk_update_task_item_results(
    conn: asyncpg.Connection,
    task_id: str,
    results: List[Dict[str, Any]],
    from_cache: bool,
) -> None:
    if not results:
        return
    arrays = _results_to_update_arrays(results, from_cache)
    await conn.execute(
        """
        UPDATE task_items AS ti
        SET
            from_cache = v.from_cache,
            tox_model_used = v.tox_model_used,
            spam_model_used = v.spam_model_used,
            is_toxic = v.is_toxic,
            toxicity_score = v.toxicity_score,
            toxicity_types = CASE
                WHEN v.toxicity_types IS NULL THEN NULL
                ELSE v.toxicity_types::jsonb
            END,
            is_spam = v.is_spam,
            spam_score = v.spam_score,
            error = v.error
        FROM (
            SELECT * FROM unnest(
                $1::boolean[],
                $2::varchar[],
                $3::varchar[],
                $4::boolean[],
                $5::real[],
                $6::text[],
                $7::boolean[],
                $8::real[],
                $9::text[],
                $10::int[]
            ) AS u(
                from_cache,
                tox_model_used,
                spam_model_used,
                is_toxic,
                toxicity_score,
                toxicity_types,
                is_spam,
                spam_score,
                error,
                item_index
            )
        ) AS v
        WHERE ti.task_id = $11 AND ti.item_index = v.item_index
        """,
        *arrays,
        task_id,
    )


async def set_task_result_pg(
    task_id: str,
    results: List[Dict[str, Any]],
    status: str = "completed",
    from_cache: bool = False,
) -> None:
    """Обновляет статус батча и записывает результаты в task_items по item_index (0-based)."""
    if not settings.database_url:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO tasks (task_id, status, completed_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (task_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        completed_at = EXCLUDED.completed_at
                    """,
                    task_id,
                    status,
                )
                await _bulk_update_task_item_results(conn, task_id, results, from_cache)
    except Exception as e:
        logger.warning("set_task_result_pg failed: %s", e)
        raise


async def set_task_failed_pg(task_id: str, error: str) -> None:
    if not settings.database_url:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'failed', completed_at = NOW()
                    WHERE task_id = $1
                    """,
                    task_id,
                )
                await conn.execute(
                    "UPDATE task_items SET error = $1 WHERE task_id = $2",
                    error,
                    task_id,
                )
    except Exception as e:
        logger.warning("set_task_failed_pg failed: %s", e)


async def get_task_pg(task_id: str) -> Optional[Dict[str, Any]]:
    """Читает задачу из Postgres. Возвращает dict с status, results (по item_index), error, user_id (первого item)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM tasks WHERE task_id = $1", task_id
        )
        if not row:
            return None
        status = row["status"]
        rows = await conn.fetch(
            """
            SELECT user_id, source, from_cache, tox_model_used, spam_model_used,
                   is_toxic, toxicity_score, toxicity_types, is_spam, spam_score, error
            FROM task_items
            WHERE task_id = $1
            ORDER BY item_index
            """,
            task_id,
        )
        user_id = rows[0][0] if rows else None
        results = []
        for r in rows:
            _, _, from_cache, tox_used, spam_used, is_toxic, tox_score, tox_types, is_spam, spam_score, err = r
            if isinstance(tox_types, str):
                try:
                    tox_types = json.loads(tox_types) if tox_types else {}
                except Exception:
                    tox_types = {}
            results.append({
                "is_toxic": bool(is_toxic) if is_toxic is not None else False,
                "toxicity_score": float(tox_score) if tox_score is not None else 0.0,
                "toxicity_types": tox_types or {},
                "tox_model_used": tox_used,
                "spam_model_used": spam_used,
                "is_spam": bool(is_spam) if is_spam is not None else False,
                "spam_score": float(spam_score) if spam_score is not None else 0.0,
                "error": err,
            })
        if status not in ("completed", "failed"):
            results = None
        error_msg = None
        if status == "failed" and rows:
            error_msg = rows[0][10] # error — последняя колонка в SELECT
        return {
            "status": status,
            "results": results,
            "error": error_msg,
            "user_id": user_id,
        }


async def delete_task_pg(task_id: str) -> None:
    """Удаляет запись задачи из Postgres (CASCADE удалит task_items)."""
    if not settings.database_url:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM tasks WHERE task_id = $1", task_id)
    except Exception as e:
        logger.warning("delete_task_pg failed: %s", e)
