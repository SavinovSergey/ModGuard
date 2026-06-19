"""Сохранение результатов бенчмарков ModGuard: JSON на прогон + CSV-сводка + Markdown-отчёт.

Один прогон описывается RunResult (условия эксперимента + метрики). save_run пишет:
  - benchmarks/results/run_<UTC>_<tag>.json — полный результат (машиночитаемо);
  - benchmarks/results/run_<UTC>_<tag>.md   — отчёт в стиле README;
  - benchmarks/results/summary.csv          — одна строка на прогон (для сравнения прогонов).

Зависимости: только стандартная библиотека.
"""
from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Стабильный порядок колонок CSV-сводки (для сравнения прогонов между собой).
CSV_COLUMNS = [
    "timestamp",
    "git_commit",
    "run_tag",
    "mode",
    "target_rate",
    "achieved_throughput",
    "throughput_steady",
    "throughput_overall",
    "e2e_p50",
    "e2e_p95",
    "e2e_p99",
    "e2e_p999",
    "e2e_max",
    "intended_p50",
    "intended_p95",
    "intended_p99",
    "intended_p999",
    "intended_max",
    "single_server_p50",
    "single_server_p95",
    "single_server_p99",
    "success_rate",
    "cache_hit_rate",
    "n_items",
    "batch_size",
    "batch_window",
    "send_workers",
    "backend_replicas",
    "pool_workers",
    "prefetch",
    "model_tox",
    "model_spam",
    "note",
]


def git_commit() -> str:
    """Короткий хэш HEAD или 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


@dataclass
class RunResult:
    """Результат одного прогона: условия эксперимента + метрики."""

    conditions: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    run_tag: str = ""
    note: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    git_commit: str = field(default_factory=git_commit)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def flat_row(self) -> Dict[str, Any]:
        """Плоская строка для CSV: известные ключи из conditions/metrics."""
        c = self.conditions
        m = self.metrics
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "run_tag": self.run_tag,
            "mode": c.get("mode"),
            "target_rate": c.get("target_rate"),
            "achieved_throughput": m.get("achieved_throughput"),
            "throughput_steady": m.get("throughput_steady"),
            "throughput_overall": m.get("throughput_overall"),
            "e2e_p50": m.get("e2e_p50"),
            "e2e_p95": m.get("e2e_p95"),
            "e2e_p99": m.get("e2e_p99"),
            "e2e_p999": m.get("e2e_p999"),
            "e2e_max": m.get("e2e_max"),
            "intended_p50": m.get("intended_p50"),
            "intended_p95": m.get("intended_p95"),
            "intended_p99": m.get("intended_p99"),
            "intended_p999": m.get("intended_p999"),
            "intended_max": m.get("intended_max"),
            "single_server_p50": m.get("single_server_p50"),
            "single_server_p95": m.get("single_server_p95"),
            "single_server_p99": m.get("single_server_p99"),
            "success_rate": m.get("success_rate"),
            "cache_hit_rate": m.get("cache_hit_rate"),
            "n_items": m.get("n_items"),
            "batch_size": c.get("batch_size"),
            "batch_window": c.get("batch_window"),
            "send_workers": c.get("send_workers"),
            "backend_replicas": c.get("backend_replicas"),
            "pool_workers": c.get("pool_workers"),
            "prefetch": c.get("prefetch"),
            "model_tox": c.get("model_tox"),
            "model_spam": c.get("model_spam"),
            "note": self.note,
        }


def _safe_tag(tag: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in (tag or "")]
    return "".join(keep).strip("-")


def _fname_timestamp(ts: str) -> str:
    return ts.replace(":", "").replace("-", "").replace("Z", "Z")


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def render_markdown(result: RunResult) -> str:
    c = result.conditions
    m = result.metrics
    lines = [
        f"# ModGuard benchmark run {result.timestamp}",
        "",
        f"- git: `{result.git_commit}`  tag: `{result.run_tag or '-'}`",
    ]
    if result.note:
        lines.append(f"- note: {result.note}")
    lines += [
        "",
        "## Условия",
        "",
        "| Параметр | Значение |",
        "|---|---|",
    ]
    for key in (
        "mode",
        "target_rate",
        "api_url",
        "val_data",
        "batch_size",
        "batch_window",
        "send_workers",
        "arrival",
        "steady_trim",
        "clear_cache",
        "backend_replicas",
        "pool_workers",
        "prefetch",
        "model_tox",
        "model_spam",
    ):
        if key in c:
            lines.append(f"| {key} | {_fmt(c[key])} |")
    lines += [
        "",
        "## Throughput",
        "",
        "| Метрика | msg/sec |",
        "|---|---|",
        f"| achieved (по send→полное окно) | {_fmt(m.get('achieved_throughput'))} |",
        f"| steady-state (плато, trim={_fmt(c.get('steady_trim'))}) | {_fmt(m.get('throughput_steady'))} |",
        f"| overall (min created→max completed) | {_fmt(m.get('throughput_overall'))} |",
        "",
        "## Latency, ms",
        "",
        "| Метрика | p50 | p95 | p99 | p99.9 | max |",
        "|---|---:|---:|---:|---:|---:|",
        "| Batch e2e (created→completed) | "
        + " | ".join(
            _fmt(m.get(k)) for k in ("e2e_p50", "e2e_p95", "e2e_p99", "e2e_p999", "e2e_max")
        )
        + " |",
        "| Intended e2e (от запланированного прибытия) | "
        + " | ".join(
            _fmt(m.get(k))
            for k in ("intended_p50", "intended_p95", "intended_p99", "intended_p999", "intended_max")
        )
        + " |",
    ]
    if m.get("single_server_p50") is not None:
        lines.append(
            "| Single-request (server-side, PG) | "
            + " | ".join(
                _fmt(m.get(k)) for k in ("single_server_p50", "single_server_p95", "single_server_p99")
            )
            + " | - | - |"
        )
    lines += [
        "",
        "## Прочее",
        "",
        f"- items: {_fmt(m.get('n_items'))}",
        f"- success rate: {_fmt(m.get('success_rate'))}",
        f"- cache hit rate: {_fmt(m.get('cache_hit_rate'))}",
        "",
    ]
    return "\n".join(lines)


def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_run(result: RunResult, results_dir: str | Path = "benchmarks/results") -> Dict[str, Path]:
    """Пишет JSON + Markdown на прогон и добавляет строку в summary.csv."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = _safe_tag(result.run_tag)
    base = f"run_{_fname_timestamp(result.timestamp)}" + (f"_{tag}" if tag else "")
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"
    csv_path = out_dir / "summary.csv"

    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")
    append_csv_row(csv_path, result.flat_row())

    return {"json": json_path, "md": md_path, "csv": csv_path}


def write_sweep_report(
    points: list[Dict[str, Any]],
    results_dir: str | Path,
    timestamp: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Сводный отчёт sweep latency-vs-load: CSV + Markdown с таблицей точек.

    points: список dict с ключами offered_rate, achieved_throughput, e2e_p50/p95/p99 (медианы по повторам).
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = f"sweep_{_fname_timestamp(ts)}"
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"

    cols = ["offered_rate", "achieved_throughput", "e2e_p50", "e2e_p95", "e2e_p99", "repeat"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for p in points:
            writer.writerow(p)

    lines = [
        f"# ModGuard latency-vs-load sweep {ts}",
        "",
    ]
    if meta:
        for k, v in meta.items():
            lines.append(f"- {k}: {_fmt(v)}")
        lines.append("")
    lines += [
        "| Offered rate, msg/s | Achieved throughput, msg/s | e2e p50, ms | e2e p95, ms | e2e p99, ms |",
        "|---:|---:|---:|---:|---:|",
    ]
    for p in points:
        lines.append(
            "| "
            + " | ".join(
                _fmt(p.get(k))
                for k in ("offered_rate", "achieved_throughput", "e2e_p50", "e2e_p95", "e2e_p99")
            )
            + " |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"csv": csv_path, "md": md_path}
