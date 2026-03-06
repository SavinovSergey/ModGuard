"""Хранилище задач асинхронной модерации (Redis или in-memory)."""
import json
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

TASK_KEY_PREFIX = "moderation:task:"


class TaskStore:
    """Хранилище задач: create_task, set_task_result, get_task."""

    def __init__(self, redis_client: Optional[Any] = None):
        self._redis = redis_client
        self._ttl = settings.task_result_ttl_seconds

    def create_task(self, task_id: str, items: List[Dict[str, Any]]) -> None:
        """Создаёт задачу со статусом queued."""
        payload = {
            "status": "queued",
            "items": items,
            "results": None,
            "error": None,
        }
        self._set(task_id, payload)

    def set_task_processing(self, task_id: str) -> None:
        """Устанавливает статус processing."""
        data = self.get_task(task_id)
        if data:
            data["status"] = "processing"
            self._set(task_id, data)

    def set_task_result(
        self,
        task_id: str,
        results: List[Dict[str, Any]],
        status: str = "completed",
    ) -> None:
        """Сохраняет результат задачи (status: completed или failed)."""
        data = self.get_task(task_id) or {"items": [], "error": None}
        data["status"] = status
        data["results"] = results
        self._set(task_id, data)

    def set_task_failed(self, task_id: str, error: str) -> None:
        """Отмечает задачу как failed с сообщением об ошибке."""
        data = self.get_task(task_id) or {"items": [], "results": None}
        data["status"] = "failed"
        data["error"] = error
        self._set(task_id, data)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает задачу: status, results (при completed), error (при failed)."""
        raw = self._get(task_id)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        return raw

    def _set(self, task_id: str, payload: Dict[str, Any]) -> None:
        if self._redis:
            try:
                key = f"{TASK_KEY_PREFIX}{task_id}"
                self._redis.setex(
                    key,
                    self._ttl,
                    json.dumps(payload, ensure_ascii=False),
                )
            except Exception as e:
                logger.warning(f"TaskStore set error: {e}")
        else:
            _memory_store[task_id] = payload

    def _get(self, task_id: str) -> Optional[Dict[str, Any]]:
        if self._redis:
            try:
                key = f"{TASK_KEY_PREFIX}{task_id}"
                raw = self._redis.get(key)
                if raw is None:
                    return None
                return json.loads(raw)
            except Exception as e:
                logger.warning(f"TaskStore get error: {e}")
                return None
        return _memory_store.get(task_id)


# In-memory fallback при отсутствии Redis (данные теряются при рестарте)
_memory_store: Dict[str, Dict[str, Any]] = {}


class MemoryTaskStore(TaskStore):
    """Task store только в памяти (для тестов или при redis_url=None)."""

    def __init__(self) -> None:
        super().__init__(redis_client=None)
