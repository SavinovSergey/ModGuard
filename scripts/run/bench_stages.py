#!/usr/bin/env python3
"""Профилирование цепочки по этапам на отдельных крупных батчах.

Отличие от capacity-дампа validate_chain: батчи отправляются **последовательно**
(один за другим, с ожиданием completed), без очереди из 33 параллельных задач.
Так видно реальное время каждого этапа на batch_size items.

Требования:
  - CHAIN_TIMING=1 на api и backend
  - docker compose с api + backend

Пример:
  CHAIN_TIMING=1 docker compose up -d --scale backend=4 api backend

  python scripts/run/bench_stages.py \\
    --val-data data/toxicity/val.parquet \\
    --batch-size 2000 --repeat 5 --warmup 1 --clear-cache

После прогона выводит:
  - клиентские метрики (POST, poll, PG e2e) по каждому батчу
  - сводку chain_timing из docker logs (--since прогона)
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import httpx

from scripts.run.summarize_chain_timing import parse_stream, summarize
from scripts.run.validate_chain import (
    load_validation_rows,
    pg_tasks_e2e_seconds,
    poll_until_done_async,
    send_batch_async,
)


async def _run_one_batch(
    client: httpx.AsyncClient,
    api_url: str,
    texts: list[str],
    poll_interval: float,
    timeout: float,
) -> dict:
    items = [(t, i) for i, t in enumerate(texts)]
    t0 = time.perf_counter()
    tids, status = await send_batch_async(client, api_url, items)
    post_ms = (time.perf_counter() - t0) * 1000
    if not tids or status != 200:
        return {
            "ok": False,
            "status": status,
            "post_ms": post_ms,
            "poll_ms": 0.0,
            "e2e_pg_ms": None,
            "n_items": len(texts),
        }

    t_poll = time.perf_counter()
    summary = await poll_until_done_async(client, api_url, tids, poll_interval, timeout)
    poll_ms = (time.perf_counter() - t_poll) * 1000

    e2e_map = pg_tasks_e2e_seconds(tids)
    e2e_pg_ms = None
    if e2e_map:
        e2e_pg_ms = max(e2e_map.values()) * 1000

    completed = all(summary["statuses"].get(t) == "completed" for t in tids)
    return {
        "ok": completed,
        "status": status,
        "task_ids": tids,
        "post_ms": post_ms,
        "poll_ms": poll_ms,
        "e2e_pg_ms": e2e_pg_ms,
        "n_items": len(texts),
    }


def _fetch_docker_logs(since: str) -> str:
    try:
        out = subprocess.run(
            ["docker", "compose", "logs", "--since", since, "api", "backend"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_root,
        )
        return out.stdout + out.stderr
    except Exception as e:
        return f"# docker logs failed: {e}\n"


def _print_client_table(rows: list[dict]) -> None:
    print("\n## Клиент + Postgres (по батчам)")
    print("| # | items | POST ms | poll ms | PG e2e ms | ok |")
    print("|---:|---:|---:|---:|---:|:---:|")
    for i, r in enumerate(rows, 1):
        e2e = f"{r['e2e_pg_ms']:.0f}" if r.get("e2e_pg_ms") is not None else "-"
        print(
            f"| {i} | {r['n_items']} | {r['post_ms']:.0f} | {r['poll_ms']:.0f} | {e2e} | {'✓' if r.get('ok') else '✗'} |"
        )
    ok_rows = [r for r in rows if r.get("ok") and r.get("e2e_pg_ms") is not None]
    if ok_rows:
        posts = [r["post_ms"] for r in ok_rows]
        polls = [r["poll_ms"] for r in ok_rows]
        e2es = [r["e2e_pg_ms"] for r in ok_rows]
        print(
            f"\nСреднее (ok): POST={sum(posts)/len(posts):.0f} ms, "
            f"poll={sum(polls)/len(polls):.0f} ms, "
            f"PG e2e={sum(e2es)/len(e2es):.0f} ms"
        )


async def _async_main(args: argparse.Namespace) -> int:
    val_path = Path(args.val_data)
    texts, _, _ = load_validation_rows(val_path, None, label_col=None, spam_label_col=None)
    if not texts:
        print("Нет текстов в val-data", file=sys.stderr)
        return 1

    batch_size = min(args.batch_size, len(texts))
    batch_texts = texts[:batch_size]

    if args.clear_cache:
        from scripts.run.clear_redis import clear_rs

        clear_rs()
        print("Кэш Redis очищен.")

    since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Старт прогона UTC {since}")
    print(f"batch_size={batch_size}, warmup={args.warmup}, repeat={args.repeat}, sequential")
    print(f"api={args.api_url}")
    print("Убедитесь: CHAIN_TIMING=1 на api/backend, MODERATION_PIPELINE=both|tox_only")
    print()

    measure_rows: list[dict] = []
    timeout = httpx.Timeout(args.send_timeout, connect=10.0)
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # health API
        try:
            hr = await client.get(f"{args.api_url.rstrip('/')}/api/v1/health")
            if hr.status_code != 200:
                print(f"API health: HTTP {hr.status_code}", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"API недоступен: {e}", file=sys.stderr)
            return 1

        total = args.warmup + args.repeat
        for i in range(total):
            is_warmup = i < args.warmup
            label = "warmup" if is_warmup else f"run {i - args.warmup + 1}/{args.repeat}"
            print(f"--- {label} ---")
            row = await _run_one_batch(
                client, args.api_url, batch_texts, args.poll_interval, args.timeout
            )
            if is_warmup:
                print(
                    f"  warmup: POST={row['post_ms']:.0f} ms, "
                    f"PG e2e={row.get('e2e_pg_ms') or 0:.0f} ms, ok={row.get('ok')}"
                )
            else:
                measure_rows.append(row)
                e2e = row.get("e2e_pg_ms")
                if e2e is not None:
                    print(
                        f"  POST={row['post_ms']:.0f} ms, poll={row['poll_ms']:.0f} ms, "
                        f"PG e2e={e2e:.0f} ms"
                    )
                else:
                    print(f"  POST={row['post_ms']:.0f} ms, poll={row['poll_ms']:.0f} ms (no PG e2e)")

    _print_client_table(measure_rows)

    print("\n## chain_timing из docker logs (--since старта прогона)")
    log_text = _fetch_docker_logs(since)
    if log_text.startswith("# docker logs failed"):
        print(log_text)
    else:
        import io

        events, classify_rows = parse_stream(io.StringIO(log_text))
        print(
            summarize(
                events,
                classify_rows,
                min_items=max(1, batch_size // 2),
                show_samples=args.show_samples,
            )
        )

    print(
        "\nПодсказка: для ручной сводки только за прогон:\n"
        f"  docker compose logs --since {since} api backend 2>&1 \\\n"
        f"    | python scripts/run/summarize_chain_timing.py --min-items {max(1, batch_size // 2)}"
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Профилирование этапов на крупных батчах (последовательно)")
    p.add_argument("--val-data", type=Path, required=True)
    p.add_argument("--api-url", type=str, default="http://localhost:8000")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--repeat", type=int, default=5, help="Измеряемых прогонов (без warmup)")
    p.add_argument("--warmup", type=int, default=1, help="Прогревов (не в таблице)")
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--poll-interval", type=float, default=0.1)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--send-timeout", type=float, default=300.0)
    p.add_argument("--show-samples", action="store_true", help="Примеры событий в summarize")
    args = p.parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
