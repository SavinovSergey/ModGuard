"""Поэтапные метки времени цепочки API → очередь → worker → Postgres/Redis.

Включение: CHAIN_TIMING=1 (или true) в окружении api/backend.
В логах ищите префикс chain_timing; поле wall — Unix time для сопоставления
перекрытия этапов между контейнерами (один wall, разные component).

Пример:
  docker compose logs api backend 2>&1 | rg chain_timing
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from app.core.config import settings

logger = logging.getLogger("modguard.chain")


def enabled() -> bool:
    return settings.chain_timing


def _tid(task_id: Optional[str]) -> str:
    return (task_id or "-")[:8]


def mark(
    component: str,
    stage: str,
    event: str,
    *,
    task_id: Optional[str] = None,
    n_items: int = 0,
    ms: Optional[float] = None,
    extra: Optional[str] = None,
) -> None:
    if not enabled():
        return
    msg = (
        f"chain_timing component={component} stage={stage} event={event} "
        f"wall={time.time():.3f} task={_tid(task_id)} n={n_items}"
    )
    if ms is not None:
        msg += f" ms={ms:.1f}"
    if extra:
        msg += f" {extra}"
    logger.info(msg)


@contextmanager
def stage(
    component: str,
    stage_name: str,
    *,
    task_id: Optional[str] = None,
    n_items: int = 0,
    extra: Optional[str] = None,
) -> Iterator[None]:
    if not enabled():
        yield
        return
    mark(component, stage_name, "start", task_id=task_id, n_items=n_items, extra=extra)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        mark(
            component,
            stage_name,
            "end",
            task_id=task_id,
            n_items=n_items,
            ms=(time.perf_counter() - t0) * 1000,
            extra=extra,
        )
