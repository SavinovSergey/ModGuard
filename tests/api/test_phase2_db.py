"""Тесты Фазы 2: слой Postgres (moderation_tasks). Требуют DATABASE_URL."""
import os
import uuid

import pytest

# Пропуск всех тестов модуля, если нет Postgres
pytest.importorskip("psycopg2")
from app.core.config import settings

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL") and not getattr(settings, "database_url", None),
    reason="DATABASE_URL not set, Phase 2 Postgres tests skipped",
)


@pytest.fixture(scope="module")
def database_url():
    url = os.environ.get("DATABASE_URL") or getattr(settings, "database_url", None)
    if not url:
        pytest.skip("DATABASE_URL not set")
    try:
        import psycopg2
        conn = psycopg2.connect(url)
        conn.close()
    except Exception as e:
        pytest.skip(f"Postgres is not reachable for DATABASE_URL ({e})")
    return url


@pytest.fixture
def task_id():
    return str(uuid.uuid4())


def test_init_db(database_url):
    """init_db создаёт таблицу без ошибок."""
    from app.core.db import init_db
    assert init_db() is True


def test_create_and_get_task_queued(database_url, task_id):
    """create_task_pg и get_task_pg: задача в статусе queued."""
    from app.core.db import create_task_pg, get_task_pg
    items = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
    create_task_pg(task_id, items, source="test")
    data = get_task_pg(task_id)
    assert data is not None
    assert data["status"] == "queued"
    assert data["results"] is None
    assert data["error"] is None


def test_set_result_and_get_task_completed(database_url, task_id):
    """set_task_result_pg и get_task_pg: статус completed и results."""
    from app.core.db import create_task_pg, set_task_result_pg, get_task_pg
    items = [{"id": "1", "text": "a"}]
    create_task_pg(task_id, items)
    results = [
        {"is_toxic": False, "toxicity_score": 0.1, "toxicity_types": {}, "tox_model_used": "regex", "spam_model_used": None},
    ]
    set_task_result_pg(task_id, results, status="completed")
    data = get_task_pg(task_id)
    assert data is not None
    assert data["status"] == "completed"
    assert data["results"] is not None
    assert len(data["results"]) == 1
    assert data["results"][0]["is_toxic"] is False
    assert data["results"][0]["toxicity_score"] == 0.1


def test_set_failed_and_get_task(database_url, task_id):
    """set_task_failed_pg и get_task_pg: статус failed и error."""
    from app.core.db import create_task_pg, set_task_failed_pg, get_task_pg
    create_task_pg(task_id, [{"text": "x"}])
    set_task_failed_pg(task_id, "Something broke")
    data = get_task_pg(task_id)
    assert data is not None
    assert data["status"] == "failed"
    assert data["error"] == "Something broke"


def test_get_task_nonexistent(database_url):
    """get_task_pg возвращает None для несуществующего task_id."""
    from app.core.db import get_task_pg
    data = get_task_pg("00000000-0000-0000-0000-000000000000")
    assert data is None
