#!/usr/bin/env python3
"""
Валидация цепочки: API → очередь → бэкенд → Postgres/кэш.

Загружает тексты из parquet, накапливает батчи (каждые 5–50 ms добавляется одно сообщение;
батч отправляется при 50 сообщениях или 3 с с момента первого), отправляет POST /classify/batch-async,
опрашивает GET /tasks/{task_id} до завершения. Часть сообщений можно отправить повторно для проверки кэша.

Запуск:
  python scripts/run/validate_chain.py --val-data data/val.parquet --max-samples 200
  python scripts/run/validate_chain.py --val-data data/val.parquet -n 500 --duplicate-ratio 0.2
  python scripts/run/validate_chain.py --val-data data/val.parquet -n 200 --duplicate-after-sec 30
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
os.chdir(_root)

import requests
from tqdm import tqdm


def count_batches(
    events: list[tuple[float, int, bool]],
    batch_size: int,
    batch_window_sec: float,
) -> int:
    """Подсчёт числа батчей по той же логике, что и в основном цикле (flush по размеру или по времени)."""
    n_batches = 0
    batch_len = 0
    batch_first_time = None
    for event_time, idx, _ in events:
        batch_len += 1
        if batch_first_time is None:
            batch_first_time = event_time
        now = event_time
        flush = batch_len >= batch_size or (now - batch_first_time) >= batch_window_sec
        if flush:
            n_batches += 1
            batch_len = 0
            batch_first_time = None
    if batch_len > 0:
        n_batches += 1
    return n_batches


def get_cache_stats_from_postgres(task_ids: list[str]) -> tuple[int, int] | None:
    """Возвращает (число из кэша, всего сообщений) по task_ids в task_items, или None при ошибке/нет БД."""
    from app.core.config import settings
    from app.core.db import get_db_connection
    if not settings.database_url or not task_ids:
        return None
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT from_cache, COUNT(*) FROM task_items WHERE task_id = ANY(%s) GROUP BY from_cache",
                    (task_ids,),
                )
                rows = cur.fetchall()
        n_cache = sum(c for from_cache, c in rows if from_cache)
        n_total = sum(c for _, c in rows)
        return (n_cache, n_total)
    except Exception as e:
        print(f"  Запрос процента из кэша: {e}")
        return None


def load_texts(path: Path, max_samples: int | None) -> list[str]:
    """Загружает колонку text из parquet (или csv). Ограничивает число строк через max_samples."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    import pandas as pd
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path, columns=["text"])
    texts = df["text"].astype(str).tolist()
    if max_samples is not None:
        texts = texts[: max_samples]
    return texts


def send_batch(api_url: str, items: list[dict], prefix: str) -> tuple[list[str], int]:
    """POST /classify/batch-async. Возвращает (список task_id для опроса, статус-код)."""
    url = f"{api_url.rstrip('/')}/api/v1/classify/batch-async"
    payload = {"items": [{"text": t} for t in items], "source": "validate_chain"}
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            return [], r.status_code
        data = r.json()
        if data.get("task_ids"):
            task_ids = list(data["task_ids"])
        else:
            task_ids = [data["task_id"]]
        return task_ids, r.status_code
    except Exception as e:
        print(f"{prefix} Ошибка запроса: {e}")
        return [], 0


def poll_until_done(api_url: str, task_ids: list[str], poll_interval: float, timeout: float) -> dict:
    """Опрашивает GET /tasks/{task_id} пока все не completed/failed или не истек timeout. Возвращает сводку."""
    url_base = f"{api_url.rstrip('/')}/api/v1/tasks"
    started = time.perf_counter()
    statuses = {tid: None for tid in task_ids}
    results_count = {tid: 0 for tid in task_ids}
    errors = {}
    while time.perf_counter() - started < timeout:
        done = True
        for tid in task_ids:
            if statuses[tid] in ("completed", "failed"):
                continue
            try:
                r = requests.get(f"{url_base}/{tid}", timeout=10)
                if r.status_code != 200:
                    errors[tid] = f"HTTP {r.status_code}"
                    statuses[tid] = "error"
                    continue
                data = r.json()
                s = data.get("status", "")
                statuses[tid] = s
                if s == "completed":
                    results_count[tid] = len(data.get("results") or [])
                elif s == "failed":
                    errors[tid] = data.get("error") or "unknown"
                else:
                    done = False
            except Exception as e:
                errors[tid] = str(e)
                done = False
        if done:
            break
        time.sleep(poll_interval)
    elapsed = time.perf_counter() - started
    return {
        "statuses": statuses,
        "results_count": results_count,
        "errors": errors,
        "elapsed": elapsed,
        "all_done": all(statuses.get(t) in ("completed", "failed", "error") for t in task_ids),
    }


def run(
    val_data: Path,
    max_samples: int | None,
    api_url: str,
    batch_size: int,
    batch_window_sec: float,
    delay_min_ms: int,
    delay_max_ms: int,
    duplicate_ratio: float,
    duplicate_after_sec: float,
    poll_interval: float,
    timeout: float,
    clear_cache: bool = False,
) -> None:
    if clear_cache:
        from scripts.run.clear_redis import clear_rs
        clear_rs()

    texts = load_texts(val_data, max_samples)
    n = len(texts)
    print(f"Загружено текстов: {n} (файл: {val_data})")
    if n == 0:
        print("Нет данных для отправки.")
        return

    # Индексы для повторной отправки (проверка кэша)
    n_dup = max(0, int(n * duplicate_ratio))
    duplicate_indices = random.sample(range(n), min(n_dup, n)) if n_dup else []
    duplicate_after_sec = max(0.0, duplicate_after_sec)

    # События: (время, индекс_текста, это_дубликат). Дубликаты ставим позже по времени,
    # чтобы они ушли в следующих батчах после первой волны — тогда кэш уже заполнен воркером.
    events = []
    t = 0.0
    for i in range(n):
        events.append((t, i, False))
        t += random.randint(delay_min_ms, delay_max_ms) / 1000.0
    t_dup = t + duplicate_after_sec
    for idx in duplicate_indices:
        events.append((t_dup, idx, True))
        t_dup += random.randint(delay_min_ms, delay_max_ms) / 1000.0
    events.sort(key=lambda x: x[0])

    n_batches = count_batches(events, batch_size, batch_window_sec)
    batch = []
    batch_first_time = None
    all_task_ids = []
    batch_times = []
    duplicate_task_ids = []

    print(f"Параметры: batch_size={batch_size}, batch_window={batch_window_sec}s, delay={delay_min_ms}-{delay_max_ms}ms")
    print(f"Дубликаты: {len(duplicate_indices)} сообщений через {duplicate_after_sec}s (проверка кэша)")
    print(f"Ожидаемое число батчей: {n_batches}")

    start_wall = time.perf_counter()
    pbar = tqdm(total=n_batches, desc="Батчи", unit="batch")
    for event_time, idx, is_dup in events:
        # Реальная задержка: ждём до момента, когда должно «прийти» это сообщение (5–50 ms между сообщениями)
        elapsed = time.perf_counter() - start_wall
        if event_time > elapsed:
            time.sleep(event_time - elapsed)
        text = texts[idx]
        batch.append((text, is_dup))
        if batch_first_time is None:
            batch_first_time = event_time
        now = event_time
        flush = (
            len(batch) >= batch_size
            or (now - batch_first_time) >= batch_window_sec
        )
        if not flush:
            continue

        items = [t for t, _ in batch]
        has_dup = any(d for _, d in batch)
        t0 = time.perf_counter()
        task_ids, status = send_batch(api_url, items, "[batch]")
        t1 = time.perf_counter()
        batch_times.append((len(items), t1 - t0, has_dup))
        if task_ids:
            all_task_ids.extend(task_ids)
            if has_dup:
                duplicate_task_ids.extend(task_ids)
        else:
            print(f"  [batch] HTTP {status}, items={len(items)}")
        pbar.update(1)
        batch = []
        batch_first_time = None

    if batch:
        items = [t for t, _ in batch]
        has_dup = any(d for _, d in batch)
        t0 = time.perf_counter()
        task_ids, status = send_batch(api_url, items, "[batch]")
        t1 = time.perf_counter()
        batch_times.append((len(items), t1 - t0, has_dup))
        if task_ids:
            all_task_ids.extend(task_ids)
            if has_dup:
                duplicate_task_ids.extend(task_ids)
        pbar.update(1)
    pbar.close()

    if not all_task_ids:
        print("Нет task_id для опроса.")
        return

    print(f"Отправлено батчей: {len(batch_times)}, всего task_id: {len(all_task_ids)}")
    print("Ожидание завершения (polling)...")
    summary = poll_until_done(api_url, all_task_ids, poll_interval, timeout)

    # Сводка
    completed = sum(1 for tid in all_task_ids if summary["statuses"].get(tid) == "completed")
    failed = sum(1 for tid in all_task_ids if summary["statuses"].get(tid) == "failed")
    total_results = sum(summary["results_count"].get(tid, 0) for tid in all_task_ids)
    print()
    print("--- Результаты ---")
    print(f"  Завершено успешно: {completed}/{len(all_task_ids)}")
    print(f"  С ошибкой:         {failed}")
    print(f"  Всего результатов: {total_results}")
    print(f"  Время опроса:      {summary['elapsed']:.2f}s")
    if summary["errors"]:
        for tid, err in list(summary["errors"].items())[:5]:
            print(f"  Ошибка {tid[:8]}...: {err}")
    if batch_times:
        avg_send = sum(t for _, t, _ in batch_times) / len(batch_times)
        print(f"  Среднее время отправки батча: {avg_send*1000:.0f}ms")

    # Процент сообщений, выданных из кэша (по Postgres task_items.from_cache)
    stats = get_cache_stats_from_postgres(all_task_ids)
    if stats is not None:
        n_cache, n_total = stats
        pct = 100.0 * n_cache / n_total if n_total else 0.0
        print(f"  Процент сообщений из кэша:   {pct:.1f}% ({n_cache}/{n_total})")
    else:
        print("  Процент из кэша: недоступен (нет DATABASE_URL или запись в task_items)")
    print("Готово.")


def main():
    p = argparse.ArgumentParser(
        description="Валидация цепочки API → очередь → бэкенд → Postgres/кэш по данным из parquet/csv",
    )
    p.add_argument("--val-data", type=Path, required=True, help="Путь к val.parquet или val.csv (колонка text)")
    p.add_argument("-n", "--max-samples", type=int, default=None, help="Макс. число строк файла для использования (по умолчанию все)")
    p.add_argument("--api-url", type=str, default="http://localhost:8000", help="Базовый URL API")
    p.add_argument("--batch-size", type=int, default=50, help="Отправлять батч при достижении N сообщений")
    p.add_argument("--batch-window", type=float, default=3.0, help="Отправлять батч при N секундах с первого сообщения в батче")
    p.add_argument("--delay-min-ms", type=int, default=5, help="Мин. задержка между добавлением сообщений в батч (ms)")
    p.add_argument("--delay-max-ms", type=int, default=50, help="Макс. задержка (ms)")
    p.add_argument("--duplicate-ratio", type=float, default=0.2, help="Доля сообщений для повторной отправки (проверка кэша)")
    p.add_argument("--duplicate-after-sec", type=float, default=10.0, help="Через сколько секунд начинать повторную отправку")
    p.add_argument("--poll-interval", type=float, default=0.2, help="Интервал опроса GET /tasks (сек)")
    p.add_argument("--timeout", type=float, default=120.0, help="Таймаут ожидания завершения всех задач (сек)")
    p.add_argument("--clear-cache", action="store_true", help="Перед прогоном очистить Redis-кэш модерации")
    args = p.parse_args()

    run(
        val_data=args.val_data,
        max_samples=args.max_samples,
        api_url=args.api_url,
        batch_size=args.batch_size,
        batch_window_sec=args.batch_window,
        delay_min_ms=args.delay_min_ms,
        delay_max_ms=args.delay_max_ms,
        duplicate_ratio=args.duplicate_ratio,
        duplicate_after_sec=args.duplicate_after_sec,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        clear_cache=args.clear_cache,
    )


if __name__ == "__main__":
    main()
