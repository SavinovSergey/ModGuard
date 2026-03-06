"""Postgres: таблица moderation_tasks и слой доступа (вариант C — полная и лёгкая запись)."""
import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS moderation_tasks (
    task_id VARCHAR(64) PRIMARY KEY,
    status VARCHAR(32) NOT NULL,
    source VARCHAR(64),
    user_id BIGINT,
    request_payload JSONB,
    response_payload JSONB,
    from_cache BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_moderation_tasks_status ON moderation_tasks(status);
CREATE INDEX IF NOT EXISTS idx_moderation_tasks_created_at ON moderation_tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_moderation_tasks_user_id ON moderation_tasks(user_id);
"""

ADD_USER_ID_COLUMN_SQL = """
ALTER TABLE moderation_tasks ADD COLUMN IF NOT EXISTS user_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_moderation_tasks_user_id ON moderation_tasks(user_id);
"""


def _get_connection():
    import psycopg2
    return psycopg2.connect(settings.database_url)


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


def init_db() -> None:
    """Создаёт таблицу при старте (если database_url задан)."""
    if not settings.database_url:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                cur.execute(ADD_USER_ID_COLUMN_SQL)
        logger.info("Postgres moderation_tasks table ready")
    except Exception as e:
        logger.warning("Postgres init_db failed: %s", e)


def create_task_pg(
    task_id: str,
    items: List[Dict[str, Any]],
    source: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    """Создаёт запись задачи со статусом queued."""
    if not settings.database_url:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO moderation_tasks (task_id, status, source, user_id, request_payload, from_cache)
                    VALUES (%s, 'queued', %s, %s, %s, FALSE)
                    ON CONFLICT (task_id) DO UPDATE SET 
                        status = 'queued', source = EXCLUDED.source, 
                        user_id = EXCLUDED.user_id, 
                        request_payload = EXCLUDED.request_payload, 
                        from_cache = FALSE
                    """,
                    (task_id, source, user_id, json.dumps({"items": items}, ensure_ascii=False)),
                )
    except Exception as e:
        logger.warning("create_task_pg failed: %s", e)
        raise


def set_task_processing_pg(task_id: str) -> None:
    if not settings.database_url:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE moderation_tasks SET status = 'processing' WHERE task_id = %s",
                    (task_id,),
                )
    except Exception as e:
        logger.warning("set_task_processing_pg failed: %s", e)


def set_task_result_pg(
    task_id: str,
    results: List[Dict[str, Any]],
    status: str = "completed",
    from_cache: bool = False,
) -> None:
    if not settings.database_url:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE moderation_tasks
                    SET status = %s, response_payload = %s, from_cache = %s, completed_at = NOW()
                    WHERE task_id = %s
                    """,
                    (status, json.dumps(results, ensure_ascii=False), from_cache, task_id),
                )
    except Exception as e:
        logger.warning("set_task_result_pg failed: %s", e)
        raise


def set_task_failed_pg(task_id: str, error: str) -> None:
    if not settings.database_url:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE moderation_tasks
                    SET status = 'failed', response_payload = %s, completed_at = NOW()
                    WHERE task_id = %s
                    """,
                    (json.dumps({"error": error}, ensure_ascii=False), task_id),
                )
    except Exception as e:
        logger.warning("set_task_failed_pg failed: %s", e)


def get_task_pg(task_id: str) -> Optional[Dict[str, Any]]:
    """Читает задачу из Postgres. Возвращает dict с status, results, error или None."""
    if not settings.database_url:
        return None
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status, response_payload, user_id FROM moderation_tasks WHERE task_id = %s",
                    (task_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                status, response_payload, user_id = row
                out = {"status": status, "results": None, "error": None, "user_id": user_id}
                if response_payload:
                    if isinstance(response_payload, str):
                        data = json.loads(response_payload)
                    else:
                        data = response_payload
                    if "error" in data:
                        out["error"] = data["error"]
                    if isinstance(data, list):
                        out["results"] = data
                    elif isinstance(data, dict) and "results" in data:
                        out["results"] = data["results"]
                    elif isinstance(data, dict):
                        out["results"] = [data]
                return out
    except Exception as e:
        logger.warning("get_task_pg failed: %s", e)
        return None
