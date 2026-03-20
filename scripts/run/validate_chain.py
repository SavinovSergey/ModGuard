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
import hashlib
import random
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
os.chdir(_root)

import requests
from tqdm import tqdm

from typing import Any, Iterable

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


def _cache_key(text: str) -> str:
    """Ключ модерационного кэша (должен совпадать с app/core/cache.py)."""
    from app.core.cache import CACHE_KEY_PREFIX

    normalized = (text or "").strip().lower()
    h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{CACHE_KEY_PREFIX}{h}"


def get_postgres_task_items_quality_stats(task_ids: list[str]) -> dict[str, Any]:
    """
    Возвращает метрики качества для task_items по task_ids:
      - n_items
      - n_invalid_* (пропуски/выход за диапазон/несогласованность model_used с флагами)
      - n_error_items
    """
    from app.core.config import settings
    from app.core.db import get_db_connection

    if not settings.database_url or not task_ids:
        return {}

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        t.task_id,
                        t.status,
                        COUNT(*) AS n_items,
                        SUM(CASE WHEN ti.is_toxic IS NULL THEN 1 ELSE 0 END) AS n_missing_toxic,
                        SUM(CASE WHEN ti.toxicity_score IS NULL OR ti.toxicity_score < 0 OR ti.toxicity_score > 1 THEN 1 ELSE 0 END) AS n_bad_toxic_score,
                        SUM(CASE
                            WHEN ti.tox_model_used IS NULL
                                 AND (ti.is_toxic IS TRUE OR COALESCE(ti.toxicity_score, 0) <> 0)
                            THEN 1 ELSE 0
                        END) AS n_bad_toxic_model_used,
                        SUM(CASE WHEN ti.is_spam IS NULL THEN 1 ELSE 0 END) AS n_missing_spam,
                        SUM(CASE WHEN ti.spam_score IS NULL OR ti.spam_score < 0 OR ti.spam_score > 1 THEN 1 ELSE 0 END) AS n_bad_spam_score,
                        SUM(CASE
                            WHEN ti.spam_model_used IS NULL
                                 AND (ti.is_spam IS TRUE OR COALESCE(ti.spam_score, 0) <> 0)
                            THEN 1 ELSE 0
                        END) AS n_bad_spam_model_used,
                        SUM(CASE WHEN ti.error IS NOT NULL THEN 1 ELSE 0 END) AS n_error_items
                    FROM tasks t
                    JOIN task_items ti ON ti.task_id = t.task_id
                    WHERE t.task_id = ANY(%s)
                    GROUP BY t.task_id, t.status
                    """,
                    (task_ids,),
                )
                rows = cur.fetchall()

        out: dict[str, Any] = {}
        for (
            task_id,
            status,
            n_items,
            n_missing_toxic,
            n_bad_toxic_score,
            n_bad_toxic_model_used,
            n_missing_spam,
            n_bad_spam_score,
            n_bad_spam_model_used,
            n_error_items,
        ) in rows:
            out[str(task_id)] = {
                "status": status,
                "n_items": int(n_items),
                "n_invalid_toxic": int(n_missing_toxic + n_bad_toxic_score + n_bad_toxic_model_used),
                "n_invalid_spam": int(n_missing_spam + n_bad_spam_score + n_bad_spam_model_used),
                "n_error_items": int(n_error_items),
            }
        return out
    except Exception as e:
        print(f"  [DB] Ошибка при проверке task_items: {e}")
        return {}


def _validate_redis_cache_for_task_items(task_ids: list[str]) -> dict[str, Any]:
    """Проверяет, что Redis содержит кэш для тех task_items, где заполнен tox_model_used/spam_model_used."""
    from app.core.config import settings
    from app.core.db import get_db_connection

    if not settings.redis_url:
        return {"enabled": False, "reason": "REDIS_URL not set"}
    if not task_ids:
        return {"enabled": False, "reason": "no task_ids"}

    try:
        import redis

        client = redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
    except Exception as e:
        return {"enabled": False, "reason": f"Redis connect failed: {e}"}

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT text, tox_model_used, spam_model_used
                    FROM task_items
                    WHERE task_id = ANY(%s) AND text IS NOT NULL AND text <> ''
                    """,
                    (task_ids,),
                )
                rows = cur.fetchall()
    except Exception as e:
        return {"enabled": False, "reason": f"DB read failed for cache check: {e}"}

    # key -> observed model types across task_items
    # (один и тот же текст может встречаться многократно в разных task_id)
    key_to_models: dict[str, dict[str, set[str]]] = {}
    for text, tox_model_used, spam_model_used in rows:
        if tox_model_used is None and spam_model_used is None:
            continue
        key = _cache_key(str(text))
        if key not in key_to_models:
            key_to_models[key] = {"tox_models": set(), "spam_models": set()}
        if tox_model_used is not None:
            key_to_models[key]["tox_models"].add(str(tox_model_used))
        if spam_model_used is not None:
            key_to_models[key]["spam_models"].add(str(spam_model_used))

    uniq_keys = list(key_to_models.keys())
    if not uniq_keys:
        return {"enabled": True, "checked_keys": 0, "found_keys": 0, "missing_keys": 0}

    try:
        values: list[Any] = client.mget(uniq_keys)
        found_keys = sum(1 for v in values if v is not None)
        missing_keys = len(uniq_keys) - found_keys

        def _cat_from_models(models: set[str]) -> str:
            if not models:
                return "none"
            if "regex" in models:
                return "regex"
            if "tfidf" in models:
                return "tfidf"
            return "other"

        # Группируем отсутствующие ключи по типам моделей.
        missing_by_tox_cat: dict[str, int] = {}
        missing_by_spam_cat: dict[str, int] = {}
        missing_examples: list[dict[str, Any]] = []
        for i, v in enumerate(values):
            if v is not None:
                continue
            key = uniq_keys[i]
            models = key_to_models.get(key) or {"tox_models": set(), "spam_models": set()}
            tox_cat = _cat_from_models(models.get("tox_models") or set())
            spam_cat = _cat_from_models(models.get("spam_models") or set())
            missing_by_tox_cat[tox_cat] = missing_by_tox_cat.get(tox_cat, 0) + 1
            missing_by_spam_cat[spam_cat] = missing_by_spam_cat.get(spam_cat, 0) + 1
            if len(missing_examples) < 5:
                # ключ целиком длинный; печатаем только последние 8 символов для удобства
                missing_examples.append(
                    {
                        "key_tail": key[-8:],
                        "tox_cat": tox_cat,
                        "spam_cat": spam_cat,
                        "tox_models": sorted(list(models.get("tox_models") or set())),
                        "spam_models": sorted(list(models.get("spam_models") or set())),
                    }
                )

        # На небольшом подмножестве проверим, что payload валиден:
        #  - json
        #  - scores в диапазоне
        #  - типы полей
        sample_indices = [i for i, v in enumerate(values) if v is not None][:50]
        sample_ok = 0
        sample_errors: list[str] = []
        for i in sample_indices:
            raw = values[i]
            try:
                payload = __import__("json").loads(raw)
                tox_score = float(payload.get("toxicity_score", 0.0))
                spam_score = float(payload.get("spam_score", 0.0))
                is_toxic = payload.get("is_toxic")
                is_spam = payload.get("is_spam")

                if not (
                    0.0 <= tox_score <= 1.0
                    and 0.0 <= spam_score <= 1.0
                    and isinstance(is_toxic, bool)
                    and isinstance(is_spam, bool)
                ):
                    raise ValueError("score/type out of range")

                if (
                    "toxicity_types" not in payload
                    or "tox_model_used" not in payload
                    or "spam_model_used" not in payload
                ):
                    raise ValueError("missing expected fields")

                tox_types = payload.get("toxicity_types", {})
                if not isinstance(tox_types, dict):
                    raise ValueError("toxicity_types is not dict")
                for _, v in tox_types.items():
                    vv = float(v)
                    if not (0.0 <= vv <= 1.0):
                        raise ValueError("toxicity_types value out of range")

                spam_model_used = payload.get("spam_model_used")
                tox_model_used = payload.get("tox_model_used")
                if tox_model_used is not None and not isinstance(tox_model_used, str):
                    raise ValueError("tox_model_used type invalid")
                if spam_model_used is not None and not isinstance(spam_model_used, str):
                    raise ValueError("spam_model_used type invalid")

                # Проверим согласованность model_used со значениями из task_items
                key = uniq_keys[i]
                models = key_to_models.get(key) or {"tox_models": set(), "spam_models": set()}
                tox_expected = models.get("tox_models") or set()
                spam_expected = models.get("spam_models") or set()
                if tox_model_used is not None and str(tox_model_used) not in tox_expected and tox_expected:
                    raise ValueError(f"tox_model_used mismatch: got={tox_model_used} expected={sorted(list(tox_expected))[:3]}")
                if spam_model_used is not None and str(spam_model_used) not in spam_expected and spam_expected:
                    raise ValueError(
                        f"spam_model_used mismatch: got={spam_model_used} expected={sorted(list(spam_expected))[:3]}"
                    )

                sample_ok += 1
            except Exception as e:
                sample_errors.append(f"{uniq_keys[i][-8:]}: {e}")

        return {
            "enabled": True,
            "checked_keys": len(uniq_keys),
            "found_keys": found_keys,
            "missing_keys": missing_keys,
            "missing_by_tox_cat": missing_by_tox_cat,
            "missing_by_spam_cat": missing_by_spam_cat,
            "missing_examples": missing_examples,
            "sample_ok": sample_ok,
            "sample_errors": sample_errors[:5],
        }
    except Exception as e:
        return {"enabled": False, "reason": f"Redis mget failed: {e}"}


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
    submitted_batches: list[dict[str, Any]] = []  # expected_items + returned task_ids

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
            submitted_batches.append({"n_items": len(items), "task_ids": list(task_ids), "has_dup": has_dup})
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
            submitted_batches.append({"n_items": len(items), "task_ids": list(task_ids), "has_dup": has_dup})
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

    # -----------------------------
    # Проверка корректности цепочки (DB + cache)
    # -----------------------------
    unique_task_ids = list(dict.fromkeys(all_task_ids))
    submitted_items_total = sum(b["n_items"] for b in submitted_batches)
    print()
    print("--- Проверка корректности ---")

    t_db0 = time.perf_counter()
    db_stats = get_postgres_task_items_quality_stats(unique_task_ids)
    t_db1 = time.perf_counter()
    if not db_stats:
        print("  [DB] Пропуск проверки (нет DATABASE_URL или ошибка/нет данных).")
    else:
        # 1) Проверка: API results_count == task_items количество для completed задач
        db_mismatches = []
        for tid in unique_task_ids:
            st = summary["statuses"].get(tid)
            if st != "completed":
                continue
            db_n = db_stats.get(tid, {}).get("n_items", 0)
            api_n = summary["results_count"].get(tid, 0)
            if int(db_n) != int(api_n):
                db_mismatches.append((tid, api_n, db_n))
        if db_mismatches:
            print(f"  [DB] Несоответствие результатов API vs task_items: {len(db_mismatches)} task_id")
            for tid, api_n, db_n in db_mismatches[:5]:
                print(f"    - {tid[:8]}... api_results={api_n} db_items={db_n}")
        else:
            print("  [DB] Соответствие: API results_count == task_items count (для completed).")

        # 2) Проверка: для каждого submitted батча сумма task_items по его task_ids == expected n_items
        batch_mismatches = []
        total_db_items = 0
        for b in submitted_batches:
            task_ids = b["task_ids"]
            expected = int(b["n_items"])
            n_db = sum(int(db_stats.get(tid, {}).get("n_items", 0)) for tid in task_ids)
            total_db_items += n_db
            if n_db != expected:
                batch_mismatches.append({"task_ids": task_ids, "expected": expected, "db_items": n_db})
        if batch_mismatches:
            print(f"  [DB] Ошибка количества items по батчам: {len(batch_mismatches)} батч(ей)")
            for m in batch_mismatches[:3]:
                print(f"    - expected={m['expected']} db_items={m['db_items']} task_ids={[t[:8]+'...' for t in m['task_ids']]}")
        else:
            print(f"  [DB] Items присутствуют: total_db_items={total_db_items}, submitted_items={submitted_items_total}")

        # 3) Проверка “адекватности” полей для completed задач
        bad_tasks = []
        for tid, stinfo in db_stats.items():
            if stinfo.get("status") != "completed":
                continue
            if int(stinfo.get("n_invalid_toxic", 0)) > 0 or int(stinfo.get("n_invalid_spam", 0)) > 0 or int(stinfo.get("n_error_items", 0)) > 0:
                bad_tasks.append((tid, stinfo))
        if bad_tasks:
            print(f"  [DB] Найдены невалидные/ошибочные результаты: {len(bad_tasks)} task_id")
            for tid, stinfo in bad_tasks[:5]:
                print(f"    - {tid[:8]}... invalid_toxic={stinfo['n_invalid_toxic']} invalid_spam={stinfo['n_invalid_spam']} error_items={stinfo['n_error_items']}")
        else:
            print("  [DB] Качество результатов: OK (completed tasks).")

    # 4) Проверка Redis кэша (опционально — если доступен)
    t_redis0 = time.perf_counter()
    cache_check = _validate_redis_cache_for_task_items(unique_task_ids)
    t_redis1 = time.perf_counter()
    if not cache_check:
        print("  [Redis] Пропуск проверки кэша.")
    else:
        if not cache_check.get("enabled"):
            print(f"  [Redis] Пропуск проверки кэша: {cache_check.get('reason')}")
        else:
            checked = cache_check.get("checked_keys", 0)
            found = cache_check.get("found_keys", 0)
            missing = cache_check.get("missing_keys", 0)
            print(f"  [Redis] Кэш ключи: checked={checked}, found={found}, missing={missing}")
            if missing > 0:
                print("  [Redis] Внимание: часть ключей кэша отсутствует — это повод углубиться.")
                missing_by_tox_cat = cache_check.get("missing_by_tox_cat") or {}
                missing_by_spam_cat = cache_check.get("missing_by_spam_cat") or {}
                if missing_by_tox_cat:
                    print(f"  [Redis] Missing by tox models: {missing_by_tox_cat}")
                if missing_by_spam_cat:
                    print(f"  [Redis] Missing by spam models: {missing_by_spam_cat}")
                examples = cache_check.get("missing_examples") or []
                if examples:
                    print("  [Redis] Missing examples (tail):")
                    for ex in examples:
                        print(
                            f"    - {ex.get('key_tail')} tox_cat={ex.get('tox_cat')} spam_cat={ex.get('spam_cat')} "
                            f"tox_models={ex.get('tox_models')} spam_models={ex.get('spam_models')}"
                        )
            if cache_check.get("sample_errors"):
                print(f"  [Redis] Ошибки в sample payload: {cache_check['sample_errors']}")

    print(f"  [Timing] DB validation: {(t_db1 - t_db0):.2f}s, Redis validation: {(t_redis1 - t_redis0):.2f}s")

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
