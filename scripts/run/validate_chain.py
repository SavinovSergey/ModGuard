#!/usr/bin/env python3
"""
Валидация цепочки: API → очередь → воркер → Postgres/кэш.

Загружает тексты из parquet/csv, отправляет батчами через POST /classify/batch-async,
опрашивает GET /tasks/{task_id} до завершения. Часть сообщений отправляется повторно
(дубликаты) для проверки кэша; в отчёте сравниваются cold-батчи (первая волна) и hot-батчи
(содержат повтор). После батч-теста — single-message latency с логом пауз опроса (poll_interval).

Метрики e2e latency и from_cache берутся из Postgres (tasks.created_at / completed_at,
task_items.from_cache). DATABASE_URL скрипта должен совпадать с тем, куда пишут API/воркер.

При наличии колонки с метками (по умолчанию label: 1/0 или true/false — класс «токсично»)
после батч-прогона считаются Precision / Recall / F1 по полю is_toxic ответа API
(и по is_spam, если задана --spam-label-col). Для корректного объединения результатов
при частичном попадании в Redis-кэш у скрипта должен быть тот же REDIS_URL, что у API.

Запуск:
  python scripts/run/validate_chain.py --val-data data/toxicity/val.parquet -n 200
  python scripts/run/validate_chain.py --val-data data/toxicity/val.parquet --batch-size 1000 --clear-cache
  python scripts/run/validate_chain.py --val-data data/toxicity/val.parquet --label-col label
  python scripts/run/validate_chain.py ... --label-col "" --spam-label-col is_spam   # только спам
  python scripts/run/validate_chain.py --val-data data/toxicity/val.parquet --single-latency-n 500
"""
import argparse
import math
import random
import sys
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import f1_score, precision_score, recall_score

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
os.chdir(_root)

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def percentiles(values: list[float], ps: tuple[int, ...] = (50, 95, 99)) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    out = {}
    for p in ps:
        k = (n - 1) * p / 100
        lo = int(k)
        hi = min(lo + 1, n - 1)
        out[f"p{p}"] = s[lo] + (k - lo) * (s[hi] - s[lo])
    return out


def _fmt_pcts(pcts: dict[str, float]) -> str:
    return ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in pcts.items())


def _cell_to_bool_label(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("", "nan", "none"):
        return None
    if s in ("0", "false", "no", "neg", "non-toxic", "clean", "ham", "not_spam"):
        return False
    if s in ("1", "true", "yes", "pos", "toxic", "spam"):
        return True
    try:
        return bool(int(float(s)))
    except ValueError:
        return None


def load_validation_rows(
    path: Path,
    max_samples: int | None,
    label_col: str | None,
    spam_label_col: str | None,
) -> tuple[list[str], list[bool | None], list[bool | None]]:
    """Тексты и опциональные метки токсичности / спама (None — строка без метки)."""
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")
    df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
    if max_samples is not None:
        df = df.iloc[:max_samples]
    texts = df["text"].astype(str).tolist()

    tox_labels: list[bool | None] = [None] * len(texts)
    if label_col and label_col in df.columns:
        tox_labels = [_cell_to_bool_label(x) for x in df[label_col].tolist()]
    elif label_col:
        print(f"  [metrics] Колонка «{label_col}» не найдена — P/R/F1 не считаются.")

    spam_labels: list[bool | None] = [None] * len(texts)
    if spam_label_col:
        if spam_label_col in df.columns:
            spam_labels = [_cell_to_bool_label(x) for x in df[spam_label_col].tolist()]
        else:
            print(f"  [metrics] Колонка «{spam_label_col}» не найдена — P/R/F1 по спаму не считаются.")

    return texts, tox_labels, spam_labels


def _moderation_cache_for_alignment():
    """Тот же Redis, что у API — для выравнивания частичного кэша (два task_id)."""
    import redis
    from app.core.cache import ModerationCache
    from app.core.config import settings

    if not settings.redis_url:
        return None
    try:
        client = redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        return ModerationCache(client)
    except Exception:
        return None


def _print_binary_metrics(name: str, y_true: list[bool], y_pred: list[bool]) -> None:
    if not y_true:
        return
    pr = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    rc = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0)
    print(f"  {name}: Precision={pr:.4f}, Recall={rc:.4f}, F1={f1:.4f} (n={len(y_true)})")


def _print_latency_bucket(title: str, values: list[float]) -> None:
    if not values:
        print(f"  {title}: нет данных")
        return
    pcts = percentiles(values)
    avg = sum(values) / len(values)
    print(
        f"  {title}: n={len(values)}, avg={avg * 1000:.0f}ms, {_fmt_pcts(pcts)}, "
        f"min={min(values) * 1000:.0f}ms, max={max(values) * 1000:.0f}ms"
    )


# ---------------------------------------------------------------------------
# API-клиент
# ---------------------------------------------------------------------------

def send_batch(api_url: str, items: list[tuple[str, int]]) -> tuple[list[str], int]:
    """items: (text, row_index) — row_index уходит в id для отладки; порядок = порядок в батче."""
    url = f"{api_url.rstrip('/')}/api/v1/classify/batch-async"
    payload = {
        "items": [{"text": t, "id": str(ri)} for t, ri in items],
        "source": "validate_chain",
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            return [], r.status_code
        data = r.json()
        tids = data.get("task_ids")
        if tids:
            return list(tids), r.status_code
        return [data["task_id"]], r.status_code
    except Exception as e:
        print(f"  [batch] Ошибка: {e}")
        return [], 0


def fetch_task_results(api_url: str, task_id: str) -> tuple[str, list[dict[str, Any]] | None]:
    """(status, results | None)."""
    url = f"{api_url.rstrip('/')}/api/v1/tasks/{task_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return "error", None
        data = r.json()
        st = data.get("status", "")
        if st != "completed":
            return st, None
        raw = data.get("results") or []
        return st, raw if isinstance(raw, list) else []
    except Exception as e:
        print(f"  [metrics] GET task {task_id[:8]}…: {e}")
        return "error", None


def merge_batch_predictions(
    block: dict[str, Any],
    results_by_tid: dict[str, list[dict[str, Any]]],
    align_cache: Any,
) -> list[tuple[int, bool, bool]] | None:
    """
    Сопоставляет результаты с индексами строк датасета.
    block: indices, items (texts), task_ids, cache_hits (на момент отправки).
    Возвращает список (row_idx, pred_is_toxic, pred_is_spam) или None, если выравнивание невозможно.
    """
    indices: list[int] = block["indices"]
    items: list[str] = block["items"]
    tids: list[str] = block["task_ids"]
    hits: list[bool] = block["cache_hits"]

    if len(indices) != len(items) or len(hits) != len(items):
        return None

    out: list[tuple[int, bool, bool]] = []

    if len(tids) == 1:
        res = results_by_tid.get(tids[0])
        if not res:
            return None
        if len(res) != len(items):
            print(f"  [metrics] task {tids[0][:8]}…: ожидалось {len(items)} результатов, пришло {len(res)}")
            return None
        for row_i, r in zip(indices, res):
            if r.get("error"):
                continue
            out.append((row_i, bool(r.get("is_toxic", False)), bool(r.get("is_spam", False))))
        return out

    if len(tids) != 2:
        return None

    if align_cache is None:
        print(
            "  [metrics] Пропуск батча: два task_id (частичный кэш), "
            "нужен REDIS_URL в окружении скрипта для выравнивания."
        )
        return None

    hit_pos = [j for j, h in enumerate(hits) if h]
    miss_pos = [j for j, h in enumerate(hits) if not h]
    res_cached = results_by_tid.get(tids[0])
    res_miss = results_by_tid.get(tids[1])
    if res_cached is None or res_miss is None:
        return None
    if len(res_cached) != len(hit_pos) or len(res_miss) != len(miss_pos):
        print(
            f"  [metrics] Пропуск батча: ожидалось cached={len(hit_pos)} miss={len(miss_pos)}, "
            f"получено {len(res_cached)} и {len(res_miss)}"
        )
        return None

    for pos, r in zip(hit_pos, res_cached):
        if r.get("error"):
            continue
        out.append((indices[pos], bool(r.get("is_toxic", False)), bool(r.get("is_spam", False))))
    for pos, r in zip(miss_pos, res_miss):
        if r.get("error"):
            continue
        out.append((indices[pos], bool(r.get("is_toxic", False)), bool(r.get("is_spam", False))))
    return out


def poll_until_done(api_url: str, task_ids: list[str], poll_interval: float, timeout: float) -> dict:
    """Опрашивает GET /tasks/{task_id} пока все не completed/failed или не истек timeout. Возвращает сводку."""
    url_base = f"{api_url.rstrip('/')}/api/v1/tasks"
    started = time.perf_counter()
    statuses: dict[str, str | None] = {tid: None for tid in task_ids}
    results_count: dict[str, int] = {tid: 0 for tid in task_ids}
    errors: dict[str, str] = {}

    while time.perf_counter() - started < timeout:
        pending = False
        for tid in task_ids:
            if statuses[tid] in ("completed", "failed", "error"):
                continue
            try:
                r = requests.get(f"{url_base}/{tid}", timeout=10)
                if r.status_code != 200:
                    errors[tid] = f"HTTP {r.status_code}"
                    statuses[tid] = "error"
                    continue
                data = r.json()
                st = data.get("status", "")
                statuses[tid] = st
                if st == "completed":
                    results_count[tid] = len(data.get("results") or [])
                elif st == "failed":
                    errors[tid] = data.get("error") or "unknown"
                else:
                    pending = True
            except Exception as e:
                errors[tid] = str(e)
                pending = True
        if not pending:
            break
        time.sleep(poll_interval)

    return {
        "statuses": statuses,
        "results_count": results_count,
        "errors": errors,
        "elapsed": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Postgres-метрики
# ---------------------------------------------------------------------------

def _pg_available() -> bool:
    from app.core.config import settings
    return bool(settings.database_url)


def pg_tasks_e2e_seconds(task_ids: list[str]) -> dict[str, float] | None:
    """task_id -> (completed_at - created_at) в секундах; только строки с обоими timestamp."""
    if not _pg_available() or not task_ids:
        return None
    from app.core.db import get_db_connection
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT task_id, created_at, completed_at
                    FROM tasks
                    WHERE task_id = ANY(%s) AND created_at IS NOT NULL AND completed_at IS NOT NULL
                    """,
                    (task_ids,),
                )
                out: dict[str, float] = {}
                for tid, ca, cpa in cur.fetchall():
                    dt = (cpa - ca).total_seconds()
                    if dt >= 0:
                        out[str(tid)] = float(dt)
                return out
    except Exception as e:
        print(f"  [PG] batch e2e: {e}")
        return None


def pg_cache_stats(task_ids: list[str]) -> tuple[int, int] | None:
    """(from_cache_count, total_count) по task_items."""
    if not _pg_available() or not task_ids:
        return None
    from app.core.db import get_db_connection
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT from_cache, COUNT(*) FROM task_items WHERE task_id = ANY(%s) GROUP BY from_cache",
                    (task_ids,),
                )
                rows = cur.fetchall()
        n_cache = sum(c for fc, c in rows if fc)
        n_total = sum(c for _, c in rows)
        return (n_cache, n_total)
    except Exception as e:
        print(f"  [PG] cache stats: {e}")
        return None


def pg_task_statuses(task_ids: list[str]) -> dict[str, str] | None:
    """task_id -> status из таблицы tasks."""
    if not _pg_available() or not task_ids:
        return None
    from app.core.db import get_db_connection
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT task_id, status FROM tasks WHERE task_id = ANY(%s)", (task_ids,))
                return {str(r[0]): r[1] for r in cur.fetchall()}
    except Exception as e:
        print(f"  [PG] task statuses: {e}")
        return None


def pg_quality_check(task_ids: list[str]) -> dict[str, dict[str, Any]] | None:
    """Проверка валидности полей task_items для completed задач."""
    if not _pg_available() or not task_ids:
        return None
    from app.core.db import get_db_connection
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        t.task_id, t.status, COUNT(*) AS n,
                        SUM(CASE WHEN ti.is_toxic IS NULL THEN 1 ELSE 0 END
                          + CASE WHEN ti.toxicity_score IS NULL OR ti.toxicity_score < 0 OR ti.toxicity_score > 1 THEN 1 ELSE 0 END
                          + CASE WHEN ti.tox_model_used IS NULL AND (ti.is_toxic IS TRUE OR COALESCE(ti.toxicity_score,0)<>0) THEN 1 ELSE 0 END
                        ) AS n_bad_tox,
                        SUM(CASE WHEN ti.is_spam IS NULL THEN 1 ELSE 0 END
                          + CASE WHEN ti.spam_score IS NULL OR ti.spam_score < 0 OR ti.spam_score > 1 THEN 1 ELSE 0 END
                          + CASE WHEN ti.spam_model_used IS NULL AND (ti.is_spam IS TRUE OR COALESCE(ti.spam_score,0)<>0) THEN 1 ELSE 0 END
                        ) AS n_bad_spam,
                        SUM(CASE WHEN ti.error IS NOT NULL THEN 1 ELSE 0 END) AS n_err
                    FROM tasks t JOIN task_items ti ON ti.task_id = t.task_id
                    WHERE t.task_id = ANY(%s)
                    GROUP BY t.task_id, t.status
                    """,
                    (task_ids,),
                )
                return {
                    str(r[0]): {"status": r[1], "n": int(r[2]), "bad_tox": int(r[3]), "bad_spam": int(r[4]), "err": int(r[5])}
                    for r in cur.fetchall()
                }
    except Exception as e:
        print(f"  [PG] quality check: {e}")
        return None


# ---------------------------------------------------------------------------
# Single-message latency
# ---------------------------------------------------------------------------

def _normalize_cache_text(text: str) -> str:
    return (text or "").strip().lower()


def _deduplicate_texts(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        key = _normalize_cache_text(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _run_single_latency(
    api_url: str,
    sample: list[str],
    poll_interval: float,
    timeout: float,
    title: str,
) -> list[float]:
    """Замеряет latency для одного сообщения через POST /classify и GET /tasks/{task_id}."""
    classify_url = f"{api_url.rstrip('/')}/api/v1/classify"
    task_url = f"{api_url.rstrip('/')}/api/v1/tasks"
    latencies: list[float] = []
    errors = 0
    poll_sleeps_per_msg: list[int] = []
    get_count_per_msg: list[int] = []

    print(f"\n--- {title} (n={len(sample)}) ---")
    for text in tqdm(sample, desc=title, unit="msg"):
        t0 = time.perf_counter()
        n_gets = 0
        n_sleeps = 0
        try:
            r = requests.post(classify_url, json={"text": text}, timeout=10)
            if r.status_code != 200:
                errors += 1
                continue
            task_id = r.json().get("task_id")
            if not task_id:
                errors += 1
                continue
            while time.perf_counter() - t0 < timeout:
                n_gets += 1
                pr = requests.get(f"{task_url}/{task_id}", timeout=10)
                if pr.status_code == 200:
                    st = pr.json().get("status")
                    if st == "completed":
                        latencies.append(time.perf_counter() - t0)
                        poll_sleeps_per_msg.append(n_sleeps)
                        get_count_per_msg.append(n_gets)
                        break
                    if st == "failed":
                        errors += 1
                        break
                time.sleep(poll_interval)
                n_sleeps += 1
            else:
                errors += 1
        except Exception:
            errors += 1

    if not latencies:
        print(f"  Нет успешных замеров. Ошибок: {errors}")
        return latencies

    pcts = percentiles(latencies)
    avg = sum(latencies) / len(latencies)
    total_sleep = sum(poll_sleeps_per_msg) * poll_interval
    avg_sleeps = sum(poll_sleeps_per_msg) / len(poll_sleeps_per_msg)
    avg_gets = sum(get_count_per_msg) / len(get_count_per_msg)
    print(f"  Успешно: {len(latencies)}/{len(sample)}, ошибок: {errors}")
    print(f"  Avg: {avg * 1000:.0f}ms  {_fmt_pcts(pcts)}")
    print(f"  Min: {min(latencies) * 1000:.0f}ms, Max: {max(latencies) * 1000:.0f}ms")
    print(
        f"  Опрос: poll_interval={poll_interval * 1000:.0f}ms; "
        f"на успешное сообщение в среднем {avg_sleeps:.2f} пауз(ы) опроса "
        f"(~{avg_sleeps * poll_interval * 1000:.0f}ms sleep суммарно), "
        f"{avg_gets:.2f} GET /tasks; за весь прогон sleep опроса ~{total_sleep * 1000:.0f}ms"
    )
    return latencies


def run_single_latency_test(api_url: str, texts: list[str], n: int, poll_interval: float, timeout: float, mode: str, dedup: bool) -> None:
    source = _deduplicate_texts(texts) if dedup else list(texts)
    if dedup and len(source) < len(texts):
        print(f"\n[Single latency] Дедупликация: {len(texts)} → {len(source)}")
    sample = random.sample(source, min(n, len(source)))

    if mode == "single":
        _run_single_latency(api_url, sample, poll_interval, timeout, "Single latency")
        return

    from scripts.run.clear_redis import clear_rs
    clear_rs()
    print("[Single latency] Кэш очищен.")

    cold = _run_single_latency(api_url, sample, poll_interval, timeout, "Cold-cache")
    hot = _run_single_latency(api_url, sample, poll_interval, timeout, "Hot-cache")
    if cold and hot:
        cold_avg = sum(cold) / len(cold)
        hot_avg = sum(hot) / len(hot)
        if hot_avg > 0:
            print(f"  Ускорение hot vs cold: x{cold_avg / hot_avg:.2f}")


# ---------------------------------------------------------------------------
# Главный прогон
# ---------------------------------------------------------------------------

def count_batches(events: list[tuple[float, int, bool]], batch_size: int, window: float) -> int:
    n = 0
    cur_len = 0
    first_t = None
    for t, _, _ in events:
        cur_len += 1
        if first_t is None:
            first_t = t
        if cur_len >= batch_size or (t - first_t) >= window:
            n += 1
            cur_len = 0
            first_t = None
    if cur_len > 0:
        n += 1
    return n


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
    single_latency_n: int = 0,
    single_latency_mode: str = "cold-hot",
    single_latency_dedup: bool = True,
    label_col: str | None = "label",
    spam_label_col: str | None = None,
) -> None:
    if clear_cache:
        from scripts.run.clear_redis import clear_rs
        clear_rs()

    texts, tox_labels, spam_labels = load_validation_rows(
        val_data, max_samples, label_col=label_col, spam_label_col=spam_label_col
    )
    n = len(texts)
    print(f"Загружено текстов: {n} (файл: {val_data})")
    if n == 0:
        return

    align_cache = _moderation_cache_for_alignment()
    if align_cache is None:
        print("  [metrics] REDIS_URL не задан или недоступен — при частичном кэше батч не попадёт в P/R/F1.")

    n_dup = max(0, int(n * duplicate_ratio))
    dup_indices = random.sample(range(n), min(n_dup, n)) if n_dup else []

    events: list[tuple[float, int, bool]] = []
    t = 0.0
    for i in range(n):
        events.append((t, i, False))
        t += random.randint(delay_min_ms, delay_max_ms) / 1000.0
    t_dup = t + max(0.0, duplicate_after_sec)
    for idx in dup_indices:
        events.append((t_dup, idx, True))
        t_dup += random.randint(delay_min_ms, delay_max_ms) / 1000.0
    events.sort(key=lambda x: x[0])

    n_batches = count_batches(events, batch_size, batch_window_sec)
    print(f"batch_size={batch_size}, window={batch_window_sec}s, delay={delay_min_ms}-{delay_max_ms}ms")
    print(f"Дубликаты: {len(dup_indices)} через {duplicate_after_sec}s")
    print(f"Ожидаемое число батчей: {n_batches}")

    # --- Отправка ---
    all_task_ids: list[str] = []
    submitted: list[dict[str, Any]] = []
    batch_send_times: list[float] = []
    batch_send_hot: list[bool] = []
    batch: list[tuple[str, int, bool]] = []
    batch_first: float | None = None

    def _flush_batch():
        nonlocal batch, batch_first
        if not batch:
            return
        has_dup = any(d for _, _, d in batch)
        items_text = [t for t, _, _ in batch]
        indices = [i for _, i, _ in batch]
        items_pairs = [(t, i) for t, i, _ in batch]
        cache_hits = (
            [align_cache.get_cached_result(t) is not None for t in items_text]
            if align_cache
            else [False] * len(items_text)
        )
        t0 = time.perf_counter()
        tids, status = send_batch(api_url, items_pairs)
        dt = time.perf_counter() - t0
        batch_send_times.append(dt)
        batch_send_hot.append(has_dup)
        if tids:
            submitted.append(
                {
                    "n_items": len(items_pairs),
                    "task_ids": list(tids),
                    "indices": indices,
                    "items": items_text,
                    "cache_hits": cache_hits,
                    "has_dup": has_dup,
                }
            )
            all_task_ids.extend(tids)
        else:
            print(f"  [batch] HTTP {status}, items={len(items_text)}")
        pbar.update(1)
        batch = []
        batch_first = None

    start_wall = time.perf_counter()
    pbar = tqdm(total=n_batches, desc="Батчи", unit="batch")
    for ev_time, idx, _is_dup in events:
        elapsed = time.perf_counter() - start_wall
        if ev_time > elapsed:
            time.sleep(ev_time - elapsed)
        batch.append((texts[idx], idx, _is_dup))
        if batch_first is None:
            batch_first = ev_time
        if len(batch) >= batch_size or (ev_time - batch_first) >= batch_window_sec:
            _flush_batch()
    _flush_batch()
    pbar.close()

    if not all_task_ids:
        print("Нет task_id для опроса.")
        return

    unique_ids = list(dict.fromkeys(all_task_ids))

    # --- Polling ---
    print(f"\nОтправлено батчей: {len(batch_send_times)}, task_id: {len(unique_ids)}")
    print("Polling...")
    summary = poll_until_done(api_url, unique_ids, poll_interval, timeout)

    completed = sum(1 for tid in unique_ids if summary["statuses"].get(tid) == "completed")
    failed = sum(1 for tid in unique_ids if summary["statuses"].get(tid) == "failed")
    total_results = sum(summary["results_count"].get(tid, 0) for tid in unique_ids)

    # --- Результаты ---
    print(f"\n--- Результаты ---")
    print(f"  Завершено: {completed}/{len(unique_ids)}, ошибок: {failed}")
    print(f"  Всего результатов: {total_results}")
    print(f"  Время polling: {summary['elapsed']:.2f}s")
    if summary["errors"]:
        for tid, err in list(summary["errors"].items())[:5]:
            print(f"  Ошибка {tid[:8]}…: {err}")

    if batch_send_times:
        avg_send = sum(batch_send_times) / len(batch_send_times)
        pcts = percentiles(batch_send_times)
        print(f"  Отправка батча (все): avg={avg_send * 1000:.0f}ms  {_fmt_pcts(pcts)}")
        cold_st = [t for t, hot in zip(batch_send_times, batch_send_hot) if not hot]
        hot_st = [t for t, hot in zip(batch_send_times, batch_send_hot) if hot]
        if hot_st:
            print("  Отправка батча cold vs hot:")
            _print_latency_bucket("    Cold (первая волна, без дубликатов в батче)", cold_st)
            _print_latency_bucket("    Hot (батч с повтором текста)", hot_st)
            if cold_st:
                ca, ha = sum(cold_st) / len(cold_st), sum(hot_st) / len(hot_st)
                if ha > 0:
                    print(f"    Отношение avg cold / avg hot: {ca / ha:.2f}x")

    # --- Классификация: P/R/F1 (нужна колонка меток в файле) ---
    if any(t is not None for t in tox_labels) or any(s is not None for s in spam_labels):
        results_by_tid: dict[str, list[dict[str, Any]]] = {}
        for tid in unique_ids:
            if summary["statuses"].get(tid) != "completed":
                continue
            _st, res = fetch_task_results(api_url, tid)
            if res is not None:
                results_by_tid[tid] = res

        merged_rows: list[tuple[int, bool, bool]] = []
        for block in submitted:
            part = merge_batch_predictions(block, results_by_tid, align_cache)
            if part:
                merged_rows.extend(part)

        print("\n--- Качество (батч, vs разметка) ---")
        if not merged_rows:
            print("  Не удалось сопоставить предсказания с батчами (проверьте completed и Redis при 2 task_id).")
        else:
            if any(t is not None for t in tox_labels):
                yt_list: list[bool] = []
                yp_list: list[bool] = []
                for row_i, pt, _ in merged_rows:
                    gt = tox_labels[row_i]
                    if gt is None:
                        continue
                    yt_list.append(gt)
                    yp_list.append(pt)
                if yt_list:
                    _print_binary_metrics("Токсичность (is_toxic)", yt_list, yp_list)
                else:
                    print("  Токсичность: нет строк с меткой в выборке предсказаний.")
            if any(s is not None for s in spam_labels):
                ys_t: list[bool] = []
                ps_t: list[bool] = []
                for row_i, _, ps in merged_rows:
                    gs = spam_labels[row_i]
                    if gs is None:
                        continue
                    ys_t.append(gs)
                    ps_t.append(ps)
                if ys_t:
                    _print_binary_metrics("Спам (is_spam)", ys_t, ps_t)
                else:
                    print("  Спам: нет строк с меткой spam в выборке предсказаний.")

    # --- Postgres-метрики ---
    e2e_map = pg_tasks_e2e_seconds(unique_ids)
    if e2e_map is None:
        print("  Batch e2e (Postgres): пропуск (нет DATABASE_URL или ошибка запроса)")
    elif not e2e_map:
        print("  Batch e2e (Postgres): нет строк tasks с created_at+completed_at по этим task_id")
    else:
        all_e2e = list(e2e_map.values())
        avg_all = sum(all_e2e) / len(all_e2e)
        p_all = percentiles(all_e2e)
        print(f"  Batch e2e (Postgres, все task_id): avg={avg_all * 1000:.0f}ms  {_fmt_pcts(p_all)}")
        print(f"    min={min(all_e2e) * 1000:.0f}ms, max={max(all_e2e) * 1000:.0f}ms")

        tid_hot: dict[str, bool] = {}
        for b in submitted:
            for tid in b["task_ids"]:
                tid_hot[str(tid)] = b["has_dup"]

        cold_e2e = [e2e_map[tid] for tid in unique_ids if tid in e2e_map and not tid_hot.get(tid, False)]
        hot_e2e = [e2e_map[tid] for tid in unique_ids if tid in e2e_map and tid_hot.get(tid, False)]

        print("  Batch e2e cold vs hot (Postgres; hot = task из батча с дубликатами):")
        _print_latency_bucket("    Cold", cold_e2e)
        _print_latency_bucket("    Hot", hot_e2e)
        if cold_e2e and hot_e2e:
            ca, ha = sum(cold_e2e) / len(cold_e2e), sum(hot_e2e) / len(hot_e2e)
            if ha > 0:
                print(f"    Отношение avg cold / avg hot: {ca / ha:.2f}x (>1 — cold дольше)")
        elif not dup_indices:
            print("    (Дубликаты не заданы — только cold.)")

    cache = pg_cache_stats(unique_ids)
    if cache:
        n_cache, n_total = cache
        pct = 100.0 * n_cache / n_total if n_total else 0.0
        print(f"  Из кэша: {pct:.1f}% ({n_cache}/{n_total})")

    # Throughput (по polling wall time)
    if total_results > 0 and summary["elapsed"] > 0:
        wall = time.perf_counter() - start_wall
        print(f"  Throughput: {total_results / wall:.0f} msg/sec (wall {wall:.1f}s)")

    # --- Проверка корректности (Postgres) ---
    db_statuses = pg_task_statuses(unique_ids)
    if db_statuses is not None:
        missing = [tid for tid in unique_ids if tid not in db_statuses]
        not_completed = {tid: st for tid, st in db_statuses.items() if st != "completed"}
        if missing:
            print(f"\n  [PG] Нет строки tasks для {len(missing)} task_id (DATABASE_URL может не совпадать с API/воркером)")
        if not_completed:
            by_st = Counter(not_completed.values())
            print(f"  [PG] Не completed: {dict(by_st)}")
            for tid, st in list(not_completed.items())[:5]:
                print(f"    - {tid[:8]}… status={st}")

    quality = pg_quality_check(unique_ids)
    if quality:
        bad = [(tid, q) for tid, q in quality.items() if q["status"] == "completed" and (q["bad_tox"] or q["bad_spam"] or q["err"])]
        if bad:
            print(f"  [PG] Невалидные результаты: {len(bad)} task_id")
            for tid, q in bad[:5]:
                print(f"    - {tid[:8]}… bad_tox={q['bad_tox']} bad_spam={q['bad_spam']} err={q['err']}")
        else:
            print("  [PG] Качество результатов: OK")

        items_total = sum(q["n"] for q in quality.values())
        submitted_total = sum(b["n_items"] for b in submitted)
        if items_total != submitted_total:
            print(f"  [PG] items в БД ({items_total}) ≠ отправлено ({submitted_total})")

    # --- Single latency ---
    if single_latency_n > 0:
        run_single_latency_test(api_url, texts, single_latency_n, poll_interval, timeout, single_latency_mode, single_latency_dedup)

    print("\nГотово.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Валидация цепочки API → очередь → воркер → Postgres/кэш")
    p.add_argument("--val-data", type=Path, required=True, help="Путь к parquet/csv (колонка text)")
    p.add_argument("-n", "--max-samples", type=int, default=None, help="Макс. строк")
    p.add_argument("--api-url", type=str, default="http://localhost:8000", help="Базовый URL API")
    p.add_argument("--batch-size", type=int, default=50, help="Размер батча")
    p.add_argument("--batch-window", type=float, default=3.0, help="Окно батча (сек)")
    p.add_argument("--delay-min-ms", type=int, default=5, help="Мин. задержка между сообщениями (мс)")
    p.add_argument("--delay-max-ms", type=int, default=50, help="Макс. задержка (мс)")
    p.add_argument("--duplicate-ratio", type=float, default=0.2, help="Доля дубликатов")
    p.add_argument("--duplicate-after-sec", type=float, default=10.0, help="Задержка перед дубликатами (сек)")
    p.add_argument("--poll-interval", type=float, default=0.2, help="Интервал polling (сек)")
    p.add_argument("--timeout", type=float, default=120.0, help="Таймаут polling (сек)")
    p.add_argument("--clear-cache", action="store_true", help="Очистить Redis-кэш перед прогоном")
    p.add_argument("--single-latency-n", type=int, default=0, help="Замер single-message latency (кол-во)")
    p.add_argument("--single-latency-mode", type=str, default="cold-hot", choices=["single", "cold-hot"])
    p.add_argument("--single-latency-no-dedup", action="store_true", help="Не дедуплицировать тексты для single latency")
    p.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Имя колонки метки токсичности (1/true — токсично). Пусто — не считать P/R/F1",
    )
    p.add_argument(
        "--spam-label-col",
        type=str,
        default=None,
        help="Опционально: колонка метки спама для P/R/F1 по is_spam",
    )
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
        single_latency_n=args.single_latency_n,
        single_latency_mode=args.single_latency_mode,
        single_latency_dedup=not args.single_latency_no_dedup,
        label_col=(args.label_col.strip() or None),
        spam_label_col=(args.spam_label_col.strip() if args.spam_label_col else None),
    )


if __name__ == "__main__":
    main()
