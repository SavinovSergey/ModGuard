#!/usr/bin/env python3
"""Сводка по chain_timing и classify-логам из docker compose (stdin или файл).

Пример (только крупные батчи, без healthcheck n=1):
  docker compose logs --since 5m api backend 2>&1 \\
    | python scripts/run/summarize_chain_timing.py --min-items 500

Показывает перцентили по этапам API и worker; tox/spam — отдельно (classify_tox/spam
или строки «Classify batch timing»).
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Iterable, List, Optional, TextIO


LINE_RE = re.compile(
    r"chain_timing component=(\w+) stage=(\S+) event=(\w+) "
    r"wall=([\d.]+) task=(\S+) n=(\d+)(?: ms=([\d.]+))?(?: hits=(\d+))?(?: path=(\S+))?"
)

CLASSIFY_RE = re.compile(
    r"Classify batch timing mode=(\w+) executor=(\w+) items=(\d+) "
    r"wall_ms=([\d.]+) tox_ms=([\d.]+) spam_ms=([\d.]+)"
)

API_STAGES = ("batch_async", "cache_lookup", "pg_create_task", "mq_publish_request")
WORKER_STAGES = (
    "task",
    "classify",
    "classify_tox",
    "classify_spam",
    "pg_result",
    "redis_cache",
    "mq_publish_result",
)


@dataclass
class Event:
    component: str
    stage: str
    event: str
    wall: float
    task: str
    n: int
    ms: Optional[float] = None


@dataclass
class ClassifyTiming:
    mode: str
    executor: str
    items: int
    wall_ms: float
    tox_ms: float
    spam_ms: float


@dataclass
class TaskSpan:
    task: str
    events: List[Event] = field(default_factory=list)

    @property
    def max_n(self) -> int:
        return max((e.n for e in self.events), default=0)


def parse_stream(stream: TextIO) -> tuple[List[Event], List[ClassifyTiming]]:
    events: List[Event] = []
    classify_rows: List[ClassifyTiming] = []
    for line in stream:
        m = LINE_RE.search(line)
        if m:
            events.append(
                Event(
                    component=m.group(1),
                    stage=m.group(2),
                    event=m.group(3),
                    wall=float(m.group(4)),
                    task=m.group(5),
                    n=int(m.group(6)),
                    ms=float(m.group(7)) if m.group(7) else None,
                )
            )
        cm = CLASSIFY_RE.search(line)
        if cm:
            classify_rows.append(
                ClassifyTiming(
                    mode=cm.group(1),
                    executor=cm.group(2),
                    items=int(cm.group(3)),
                    wall_ms=float(cm.group(4)),
                    tox_ms=float(cm.group(5)),
                    spam_ms=float(cm.group(6)),
                )
            )
    return events, classify_rows


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (k - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _stats(values: List[float]) -> Optional[dict[str, float | int]]:
    if not values:
        return None
    s = sorted(values)
    return {
        "n": len(s),
        "min": s[0],
        "p50": _percentile(s, 50),
        "p95": _percentile(s, 95),
        "avg": sum(s) / len(s),
        "max": s[-1],
    }


def _fmt_stats(st: dict[str, float | int]) -> str:
    return (
        f"n={int(st['n'])} "
        f"min={st['min']:.0f} p50={st['p50']:.0f} avg={st['avg']:.0f} "
        f"p95={st['p95']:.0f} max={st['max']:.0f} ms"
    )


def stage_durations(
    events: Iterable[Event],
    component: str,
    stage: str,
    min_items: int = 0,
) -> List[float]:
    return [
        e.ms
        for e in events
        if e.component == component
        and e.stage == stage
        and e.event == "end"
        and e.ms is not None
        and e.n >= min_items
    ]


def summarize(
    events: List[Event],
    classify_rows: List[ClassifyTiming],
    *,
    min_items: int = 0,
    show_samples: bool = False,
) -> str:
    lines: List[str] = []
    if min_items > 0:
        lines.append(f"Фильтр: n_items >= {min_items} (healthcheck и мелкие батчи отброшены)")
        lines.append("")

    api_ends = [
        e
        for e in events
        if e.component == "api" and e.stage == "batch_async" and e.event == "end" and e.n >= min_items
    ]
    if api_ends:
        durs = [e.ms for e in api_ends if e.ms is not None]
        st = _stats(durs)
        lines.append(f"## API (batch_async, n>={min_items})")
        if st:
            lines.append(f"  POST handler total: {_fmt_stats(st)}")
        for stage in API_STAGES:
            if stage == "batch_async":
                continue
            sd = stage_durations(events, "api", stage, min_items)
            stg = _stats(sd)
            if stg:
                lines.append(f"  api {stage}: {_fmt_stats(stg)}")
        lines.append("")

    worker_stages_present = False
    wlines: List[str] = []
    for stage in WORKER_STAGES:
        sd = stage_durations(events, "worker", stage, min_items)
        stg = _stats(sd)
        if stg:
            worker_stages_present = True
            wlines.append(f"  worker {stage}: {_fmt_stats(stg)}")

    if worker_stages_present:
        lines.append(f"## Worker (chain_timing, n>={min_items})")
        lines.extend(wlines)
        by_task: DefaultDict[str, List[Event]] = defaultdict(list)
        for e in events:
            if e.task != "-":
                by_task[e.task].append(e)
        big_tasks = [t for t, evs in by_task.items() if any(x.component == "worker" for x in evs) and max(x.n for x in evs) >= min_items]
        if big_tasks:
            points: List[tuple[float, int]] = []
            for t in big_tasks:
                for e in by_task[t]:
                    if e.component == "worker" and e.stage == "mq_consume" and e.event == "start":
                        points.append((e.wall, 1))
                    if e.component == "worker" and e.stage == "task" and e.event == "end" and e.n >= min_items:
                        points.append((e.wall, -1))
            cur = peak = 0
            for _, delta in sorted(points):
                cur += delta
                peak = max(peak, cur)
            lines.append(f"  peak concurrent large batches (approx): {peak}")
        lines.append("")

    big_classify = [c for c in classify_rows if c.items >= min_items]
    if big_classify:
        lines.append(f"## Worker classify (лог ClassificationService, items>={min_items})")
        for label, getter in (
            ("classify wall", lambda c: c.wall_ms),
            ("classify tox", lambda c: c.tox_ms),
            ("classify spam", lambda c: c.spam_ms),
            ("max(tox,spam)", lambda c: max(c.tox_ms, c.spam_ms)),
        ):
            st = _stats([getter(c) for c in big_classify])
            if st:
                lines.append(f"  {label}: {_fmt_stats(st)}")
        modes = sorted({c.mode for c in big_classify})
        execs = sorted({c.executor for c in big_classify})
        lines.append(f"  modes={modes} executors={execs}")
        lines.append("")

    if not lines:
        return "Нет данных chain_timing / classify timing с заданным фильтром.\n"

    if show_samples:
        lines.append("## Примеры chain_timing (первые 10 end, n>={})".format(min_items))
        shown = 0
        for e in events:
            if e.event != "end" or e.ms is None or e.n < min_items:
                continue
            lines.append(
                f"  {e.component:6} {e.stage:18} task={e.task} n={e.n} ms={e.ms:.1f}"
            )
            shown += 1
            if shown >= 10:
                break

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Сводка chain_timing из логов docker compose")
    p.add_argument(
        "--min-items",
        type=int,
        default=100,
        help="Учитывать только события с n_items >= N (отсечь healthcheck n=1)",
    )
    p.add_argument(
        "--samples",
        action="store_true",
        help="Показать примеры отдельных событий",
    )
    args = p.parse_args()

    events, classify_rows = parse_stream(sys.stdin)
    if not events and not classify_rows:
        print("Нет строк chain_timing / Classify batch timing во входе.", file=sys.stderr)
        sys.exit(1)

    print(summarize(events, classify_rows, min_items=args.min_items, show_samples=args.samples))


if __name__ == "__main__":
    main()
