"""Postgres: таблицы tasks (батч) и task_items (по одному сообщению), слой доступа."""
import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

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
    """Создаёт таблицы tasks и task_items, если их ещё нет (данные не удаляются)."""
    if not settings.database_url:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLES_SQL)
        logger.info("Postgres: tasks and task_items ready")
    except Exception as e:
        logger.warning("Postgres init_db failed: %s", e)


def create_task_pg(
    task_id: str,
    items: List[Dict[str, Any]],
    source: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    """Создаёт запись батча в tasks и по одной строке в task_items на каждый item (item_index 0-based)."""
    if not settings.database_url:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tasks (task_id, status)
                    VALUES (%s, 'queued')
                    ON CONFLICT (task_id) DO UPDATE SET status = 'queued'
                    """,
                    (task_id,),
                )
                for i, item in enumerate(items):
                    text = item.get("text", "")
                    cur.execute(
                        """
                        INSERT INTO task_items (task_id, item_index, user_id, source, text, from_cache)
                        VALUES (%s, %s, %s, %s, %s, FALSE)
                        ON CONFLICT (task_id, item_index) DO UPDATE SET
                            user_id = EXCLUDED.user_id,
                            source = EXCLUDED.source,
                            text = EXCLUDED.text,
                            from_cache = FALSE
                        """,
                        (task_id, i, user_id, source, text),
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
                    "UPDATE tasks SET status = 'processing' WHERE task_id = %s",
                    (task_id,),
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


def set_task_result_pg(
    task_id: str,
    results: List[Dict[str, Any]],
    status: str = "completed",
    from_cache: bool = False,
) -> None:
    """Обновляет статус батча и записывает результаты в task_items по item_index (0-based)."""
    if not settings.database_url:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET status = %s, completed_at = NOW()
                    WHERE task_id = %s
                    """,
                    (status, task_id),
                )
                for i, r in enumerate(results):
                    row = _result_to_row(r, from_cache)
                    cur.execute(
                        """
                        UPDATE task_items
                        SET from_cache = %s, tox_model_used = %s, spam_model_used = %s,
                            is_toxic = %s, toxicity_score = %s, toxicity_types = %s,
                            is_spam = %s, spam_score = %s, error = %s
                        WHERE task_id = %s AND item_index = %s
                        """,
                        (*row, task_id, i),
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
                    UPDATE tasks
                    SET status = 'failed', completed_at = NOW()
                    WHERE task_id = %s
                    """,
                    (task_id,),
                )
                cur.execute(
                    "UPDATE task_items SET error = %s WHERE task_id = %s",
                    (error, task_id),
                )
    except Exception as e:
        logger.warning("set_task_failed_pg failed: %s", e)


def get_task_pg(task_id: str) -> Optional[Dict[str, Any]]:
    """Читает задачу из Postgres. Возвращает dict с status, results (по item_index), error, user_id (первого item)."""
    if not settings.database_url:
        return None
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM tasks WHERE task_id = %s",
                    (task_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                status = row[0]
                cur.execute(
                    """
                    SELECT user_id, source, from_cache, tox_model_used, spam_model_used,
                           is_toxic, toxicity_score, toxicity_types, is_spam, spam_score, error
                    FROM task_items
                    WHERE task_id = %s
                    ORDER BY item_index
                    """,
                    (task_id,),
                )
                rows = cur.fetchall()
                user_id = rows[0][0] if rows else None
                results = []
                for r in rows:
                    _, _, from_cache, tox_used, spam_used, is_toxic, tox_score, tox_types, is_spam, spam_score, err = r
                    if isinstance(tox_types, str):
                        try:
                            tox_types = json.loads(tox_types) if tox_types else {}
                        except Exception:
                            tox_types = {}
                    # Пока задача не completed, поля результата в task_items могут быть NULL
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
                # Для queued/processing не возвращаем results (как раньше response_payload был пуст)
                if status not in ("completed", "failed"):
                    results = None
                error_msg = None
                if status == "failed" and rows:
                    error_msg = rows[0][10]  # error — последняя колонка в SELECT
                return {
                    "status": status,
                    "results": results,
                    "error": error_msg,
                    "user_id": user_id,
                }
    except Exception as e:
        logger.warning("get_task_pg failed: %s", e)
        return None


def delete_task_pg(task_id: str) -> None:
    """Удаляет запись задачи из Postgres (CASCADE удалит task_items)."""
    if not settings.database_url:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tasks WHERE task_id = %s", (task_id,))
    except Exception as e:
        logger.warning("delete_task_pg failed: %s", e)
