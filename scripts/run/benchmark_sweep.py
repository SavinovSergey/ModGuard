"""Sweep latency-vs-load для ModGuard: строит кривую насыщения.

Шаги:
  1. Capacity-probe (дамп, target_rate=0) -> измеренный потолок C (steady throughput).
  2. Open-loop прогоны при lambda = долях C (по умолчанию 0.3/0.5/0.7/0.85/0.95),
     каждая точка повторяется --repeat раз, берём медиану метрик.
  3. Сводка benchmarks/results/sweep_<ts>.{csv,md}: offered_rate / achieved_throughput /
     p50 / p95 / p99 — видно «колено» кривой.

Тонкая обвязка над run() из validate_chain (логика не дублируется).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

from scripts.run import bench_results
from scripts.run.validate_chain import run


def _median(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return statistics.median(nums) if nums else None


def _capacity(args) -> float | None:
    print("=" * 70)
    print("CAPACITY PROBE (дамп, target_rate=0)")
    print("=" * 70)
    res = run(
        val_data=args.val_data,
        max_samples=args.capacity_n,
        api_url=args.api_url,
        batch_size=args.batch_size,
        batch_window_sec=args.batch_window,
        delay_min_ms=0,
        delay_max_ms=0,
        duplicate_ratio=0.0,
        duplicate_after_sec=0.0,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        clear_cache=True,
        label_col=None,
        spam_label_col=None,
        send_workers=args.send_workers,
        send_timeout=args.send_timeout,
        target_rate=0.0,
        steady_trim=args.steady_trim,
        results_dir=args.results_dir,
        run_tag="sweep-capacity",
        note="capacity probe for sweep",
        backend_replicas=args.backend_replicas,
        pool_workers=args.pool_workers,
        prefetch=args.prefetch,
    )
    if not res:
        return None
    m = res["metrics"]
    return m.get("throughput_steady") or m.get("throughput_overall")


def _load_point(args, offered_rate: float, frac: float, rep: int) -> dict | None:
    print("-" * 70)
    print(f"LOAD POINT frac={frac:.2f} offered={offered_rate:.0f} msg/s rep={rep + 1}/{args.repeat}")
    print("-" * 70)
    res = run(
        val_data=args.val_data,
        max_samples=args.load_n,
        api_url=args.api_url,
        batch_size=args.batch_size,
        batch_window_sec=args.batch_window,
        delay_min_ms=0,
        delay_max_ms=0,
        duplicate_ratio=0.0,
        duplicate_after_sec=0.0,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        clear_cache=True,
        label_col=None,
        spam_label_col=None,
        send_workers=args.send_workers,
        send_timeout=args.send_timeout,
        target_rate=offered_rate,
        arrival=args.arrival,
        steady_trim=args.steady_trim,
        results_dir=args.results_dir,
        run_tag=f"sweep-load-{frac:.2f}-r{rep + 1}",
        note=f"sweep load point frac={frac:.2f}",
        backend_replicas=args.backend_replicas,
        pool_workers=args.pool_workers,
        prefetch=args.prefetch,
    )
    return res["metrics"] if res else None


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep latency-vs-load для ModGuard")
    p.add_argument("--val-data", type=Path, required=True)
    p.add_argument("--api-url", type=str, default="http://localhost:8000")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--batch-window", type=float, default=3.0)
    p.add_argument("--poll-interval", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--send-workers", type=int, default=32)
    p.add_argument("--send-timeout", type=float, default=300.0)
    p.add_argument("--arrival", type=str, default="uniform", choices=["uniform", "poisson"])
    p.add_argument("--steady-trim", type=float, default=0.1)
    p.add_argument("--capacity-n", type=int, default=2000, help="Сэмплов для capacity-probe")
    p.add_argument("--load-n", type=int, default=1500, help="Сэмплов на одну точку нагрузки")
    p.add_argument(
        "--fracs",
        type=str,
        default="0.3,0.5,0.7,0.85,0.95",
        help="Доли от capacity для offered rate",
    )
    p.add_argument("--repeat", type=int, default=3, help="Повторов на точку (берётся медиана)")
    p.add_argument("--capacity", type=float, default=0.0, help="Задать C вручную (пропустить probe)")
    p.add_argument("--results-dir", type=str, default="benchmarks/results")
    p.add_argument("--backend-replicas", type=int, default=0)
    p.add_argument("--pool-workers", type=int, default=0)
    p.add_argument("--prefetch", type=int, default=0)
    args = p.parse_args()

    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]

    capacity = args.capacity if args.capacity > 0 else _capacity(args)
    if not capacity or capacity <= 0:
        print("Не удалось определить capacity (C). Прерывание.")
        sys.exit(1)
    print(f"\nCapacity C = {capacity:.0f} msg/sec\n")

    points: list[dict] = []
    for frac in fracs:
        offered = frac * capacity
        achieved_reps: list[float | None] = []
        p50_reps: list[float | None] = []
        p95_reps: list[float | None] = []
        p99_reps: list[float | None] = []
        for rep in range(args.repeat):
            m = _load_point(args, offered, frac, rep)
            if not m:
                continue
            achieved_reps.append(m.get("achieved_throughput"))
            p50_reps.append(m.get("intended_p50") or m.get("e2e_p50"))
            p95_reps.append(m.get("intended_p95") or m.get("e2e_p95"))
            p99_reps.append(m.get("intended_p99") or m.get("e2e_p99"))
        points.append(
            {
                "offered_rate": round(offered, 1),
                "achieved_throughput": _median(achieved_reps),
                "e2e_p50": _median(p50_reps),
                "e2e_p95": _median(p95_reps),
                "e2e_p99": _median(p99_reps),
                "repeat": args.repeat,
            }
        )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    paths = bench_results.write_sweep_report(
        points,
        args.results_dir,
        timestamp=ts,
        meta={
            "capacity_C": capacity,
            "arrival": args.arrival,
            "load_n": args.load_n,
            "repeat": args.repeat,
            "backend_replicas": args.backend_replicas or None,
            "pool_workers": args.pool_workers or None,
        },
    )
    print(f"\nSweep сохранён: {paths['csv']}")
    print(f"  Markdown: {paths['md']}")


if __name__ == "__main__":
    main()
