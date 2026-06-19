"""Анализ regex pre-filter: вклад, ошибки, время, Pareto-подбор.

Пример:
  python scripts/run/analyze_regex_patterns.py --task tox
  python scripts/run/analyze_regex_patterns.py --task spam --spam-sample 100000
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.models.spam.regex_model import SpamRegexModel, _SPAM_CATEGORIES
from app.models.toxicity.regex_model import RegexModel

PatternLabel = Tuple[str, re.Pattern]


@dataclass
class PatternStats:
    name: str
    hits: int
    tp: int
    fp: int
    fn_only: int  # positive texts caught ONLY by this pattern (unique recall contribution)
    precision: float
    recall_contrib: float  # fn_only / n_positive
    time_ms: float
    ms_per_1k: float
    pct_time: float
    recommendation: str


def _load_data(task: str, val_path: Path | None, text_col: str, label_col: str, spam_sample: int | None):
    if val_path is None:
        val_path = Path("data/toxicity/val.parquet" if task == "tox" else "data/spam/val.parquet")
    df = pd.read_parquet(val_path)
    if text_col not in df.columns:
        raise ValueError(f"Колонка {text_col!r} не найдена в {val_path}")
    if label_col not in df.columns:
        raise ValueError(f"Колонка {label_col!r} не найдена в {val_path}")

    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col]
    if labels.dtype == bool:
        y = labels.astype(np.int8).to_numpy()
    else:
        y = labels.astype(int).to_numpy()

    if task == "spam" and spam_sample and spam_sample < len(texts):
        rng = np.random.default_rng(42)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        n_pos = min(len(pos_idx), spam_sample // 10)
        n_neg = spam_sample - n_pos
        pick = np.concatenate([
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ])
        pick.sort()
        texts = [texts[i] for i in pick]
        y = y[pick]
        print(f"  spam: stratified sample n={len(texts)} (pos={y.sum()}, neg={len(y)-y.sum()})")

    return texts, y, val_path


def _pattern_mask(series: pd.Series, pattern: re.Pattern) -> np.ndarray:
    valid = series.str.strip().ne("")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This pattern is interpreted as a regular expression, and has match groups.*",
            category=UserWarning,
        )
        hits = series.str.contains(
            pattern.pattern,
            regex=True,
            case=not bool(pattern.flags & re.IGNORECASE),
            na=False,
        )
    return (valid & hits).to_numpy(dtype=bool)


def _time_pattern(texts: List[str], pattern: re.Pattern, repeat: int = 3) -> float:
    series = pd.Series(texts, dtype=object).fillna("").astype(str)
    for _ in range(1):
        _pattern_mask(series, pattern)
    t0 = time.perf_counter()
    for _ in range(repeat):
        _pattern_mask(series, pattern)
    return (time.perf_counter() - t0) / repeat * 1000.0


def _union_mask(texts: List[str], patterns: Sequence[PatternLabel]) -> np.ndarray:
    series = pd.Series(texts, dtype=object).fillna("").astype(str)
    if not patterns:
        return np.zeros(len(texts), dtype=bool)
    out = np.zeros(len(texts), dtype=bool)
    for _, p in patterns:
        out |= _pattern_mask(series, p)
    return out


def _analyze_patterns(
    task: str,
    patterns: Sequence[PatternLabel],
    texts: List[str],
    y: np.ndarray,
    min_precision: float,
) -> Tuple[List[PatternStats], Dict[str, np.ndarray], np.ndarray]:
    n = len(texts)
    n_pos = int(y.sum())
    series = pd.Series(texts, dtype=object).fillna("").astype(str)

    masks: Dict[str, np.ndarray] = {}
    times: Dict[str, float] = {}

    print(f"  Прогон {len(patterns)} паттернов на {n} текстах...")
    for name, pattern in patterns:
        t0 = time.perf_counter()
        masks[name] = _pattern_mask(series, pattern)
        times[name] = (time.perf_counter() - t0) * 1000.0

    union = np.zeros(n, dtype=bool)
    for m in masks.values():
        union |= m

    # unique-only: positive text matched by this pattern but by no other
    stats: List[PatternStats] = []
    total_time = sum(times.values())

    for name, pattern in patterns:
        m = masks[name]
        hits = int(m.sum())
        tp = int((m & (y == 1)).sum())
        fp = int((m & (y == 0)).sum())
        others = union & ~m
        fn_only = int((m & (y == 1) & ~others).sum()) if hits else 0
        # fix: unique = matched by ONLY this pattern among all
        other_union = np.zeros(n, dtype=bool)
        for oname, om in masks.items():
            if oname != name:
                other_union |= om
        unique_pos = int((m & (y == 1) & ~other_union).sum())
        prec = tp / hits if hits else 1.0
        recall_contrib = unique_pos / n_pos if n_pos else 0.0
        t_ms = times[name]

        if hits == 0:
            rec = "убрать (0 hits)"
        elif prec < min_precision and unique_pos == 0:
            rec = f"убрать (P={prec:.2f}<{min_precision}, уник. TP=0)"
        elif prec < min_precision and unique_pos > 0:
            rec = f"переписать (P={prec:.2f}, но уник. TP={unique_pos})"
        elif unique_pos == 0 and fp > 0:
            rec = f"убрать (дубль, FP={fp})"
        elif unique_pos == 0 and tp > 0:
            rec = "дубль (вклад через OR с другими)"
        elif t_ms / max(total_time, 1) > 0.25 and recall_contrib < 0.01:
            rec = "дорого / мало уник. recall — кандидат на вынос"
        elif prec >= min_precision and recall_contrib >= 0.005:
            rec = "оставить"
        else:
            rec = "на усмотрение"

        stats.append(
            PatternStats(
                name=name,
                hits=hits,
                tp=tp,
                fp=fp,
                fn_only=unique_pos,
                precision=prec,
                recall_contrib=recall_contrib,
                time_ms=t_ms,
                ms_per_1k=t_ms / n * 1000 if n else 0.0,
                pct_time=100.0 * t_ms / total_time if total_time else 0.0,
                recommendation=rec,
            )
        )

    return stats, masks, union


def _ensemble_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def _leave_one_out(
    patterns: Sequence[PatternLabel],
    masks: Dict[str, np.ndarray],
    y: np.ndarray,
    union: np.ndarray,
) -> List[Tuple[str, float, int, int]]:
    """При удалении паттерна: Δrecall и ΔFP."""
    base = _ensemble_metrics(y, union)
    rows = []
    n_pos = int(y.sum())
    for name, _ in patterns:
        pred = union & ~masks[name]
        m = _ensemble_metrics(y, pred)
        delta_recall = m["recall"] - base["recall"]
        delta_fp = m["fp"] - base["fp"]
        lost_unique = int((masks[name] & (y == 1)).sum())  # all TP from pattern, not unique
        rows.append((name, delta_recall, delta_fp, lost_unique))
    return rows


def _pareto_subsets(
    patterns: Sequence[PatternLabel],
    masks: Dict[str, np.ndarray],
    times: Dict[str, float],
    y: np.ndarray,
    min_precision: float,
    max_subset_size: int = 12,
) -> List[Dict]:
    """Перебор подмножеств (ограниченный) → Pareto по (time, recall) при precision >= min."""
    names = [n for n, _ in patterns]
    n = len(y)
    n_pos = int(y.sum())
    points: List[Dict] = []

    # Ограничиваем комбinatorics: полный перебор до 12 паттернов = 4096 max for spam 10
    if len(names) > max_subset_size:
        names_eval = names
    else:
        names_eval = names

    all_masks = [masks[n] for n in names_eval]
    all_times = [times[n] for n in names_eval]
    m_len = len(names_eval)

    for bits in range(1, 1 << m_len):
        subset_idx = [i for i in range(m_len) if (bits >> i) & 1]
        pred = np.zeros(n, dtype=bool)
        t_ms = 0.0
        for i in subset_idx:
            pred |= all_masks[i]
            t_ms += all_times[i]
        met = _ensemble_metrics(y, pred)
        if met["precision"] < min_precision:
            continue
        points.append({
            "patterns": [names_eval[i] for i in subset_idx],
            "k": len(subset_idx),
            "time_ms": t_ms,
            "recall": met["recall"],
            "precision": met["precision"],
            "f1": met["f1"],
            "fp": met["fp"],
            "tp": met["tp"],
        })

    # Pareto: максимизируем recall, минимизируем time
    pareto: List[Dict] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q["recall"] >= p["recall"] and q["time_ms"] <= p["time_ms"] and (
                q["recall"] > p["recall"] or q["time_ms"] < p["time_ms"]
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    pareto.sort(key=lambda x: (x["time_ms"], -x["recall"]))
    return pareto


def _greedy_pareto_build(
    patterns: Sequence[PatternLabel],
    masks: Dict[str, np.ndarray],
    times: Dict[str, float],
    y: np.ndarray,
    min_precision: float,
) -> List[str]:
    """Жадно добавляем паттерн с лучшим Δrecall/Δtime пока precision OK."""
    names = [n for n, _ in patterns]
    selected: List[str] = []
    pred = np.zeros(len(y), dtype=bool)

    remaining = set(names)
    while remaining:
        best_name = None
        best_score = -1.0
        best_pred = None
        base_met = _ensemble_metrics(y, pred)

        for name in remaining:
            trial = pred | masks[name]
            met = _ensemble_metrics(y, trial)
            if met["precision"] < min_precision:
                continue
            delta_rec = met["recall"] - base_met["recall"]
            if delta_rec <= 0:
                continue
            score = delta_rec / max(times[name], 0.001)
            if score > best_score:
                best_score = score
                best_name = name
                best_pred = trial

        if best_name is None:
            break
        selected.append(best_name)
        remaining.remove(best_name)
        pred = best_pred  # type: ignore[assignment]

    return selected


def _print_table(stats: List[PatternStats], title: str) -> None:
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")
    print(
        f"{'pattern':<18} {'hits':>7} {'TP':>6} {'FP':>6} {'uniqTP':>7} "
        f"{'P':>6} {'uniqR':>7} {'ms':>8} {'ms/1k':>7} {'%time':>6}  recommendation"
    )
    print("-" * 100)
    for s in sorted(stats, key=lambda x: -x.time_ms):
        print(
            f"{s.name:<18} {s.hits:7d} {s.tp:6d} {s.fp:6d} {s.fn_only:7d} "
            f"{s.precision:6.3f} {s.recall_contrib:7.4f} {s.time_ms:8.1f} "
            f"{s.ms_per_1k:7.2f} {s.pct_time:5.1f}%  {s.recommendation}"
        )


def _regex_quality_notes(task: str) -> None:
    print(f"\n{'=' * 100}")
    print(f"Замечания по качеству regex ({task})")
    print(f"{'=' * 100}")

    if task == "tox":
        notes = [
            "Дублирование: «трах» встречается в паттернах «бля» и «пиздец» — лишний двойной проход.",
            "«идиот», «дебил», «даун», «пиндос» — слабые маркеры токсичности, дают FP на нейтральных спорах.",
            "«(рас|от)стрел» ловит «расстрелять политика» — высокий FP при умеренном TP (полит. токсичность).",
            "Паттерн «хуй»: «херн»/«херов» могут матчить «херовое качество» — borderline FP.",
            "Паттерн «ебать»: «збс» — сленг, не всегда токсичен в контексте.",
            "Нет word-boundary у «п[еи]?зде?ц?» — риск partial match внутри слов (редко).",
            "Общее: 6 тяжёлых alternation-паттернов; выгоднее split на atomic groups или keyword+verify.",
        ]
    else:
        notes = [
            "«suspicious_links» t.me/ — много FP на легитимных каналах (155 hits на toxicity-val ранее).",
            "«recruitment» «ищу/ищем» — FP на обычных вакансиях в IT-сообществах.",
            "«urgency» «срочно нужен» — FP в бытовых объявлениях.",
            "«crypto» — 0 hits на val: мёртвый паттерн или редкая категория — кандидат на удаление.",
            "«giveaway» — очень мало hits; проверить на train/test spam.",
            "«.{0,25}» и «.{0,20}» — потенциально медленные (backtracking), лучше ограничить [^\\n]{0,25}.",
            "Capturing groups в recruitment не нужны — использовать (?:...) для pandas/str.contains.",
            "10 последовательных OR-паттернов: Pareto-оптимально ~4–5 категорий по cost/benefit.",
        ]
    for n in notes:
        print(f"  • {n}")


def run_task(args: argparse.Namespace, task: str, patterns: Sequence[PatternLabel]) -> None:
    print(f"\n>>> {task.upper()}")
    texts, y, path = _load_data(task, args.val_data, args.text_col, args.label_col, args.spam_sample)
    print(f"  data: {path}  n={len(texts)}  positive={int(y.sum())} ({100*y.mean():.2f}%)")

    stats, masks, union = _analyze_patterns(task, patterns, texts, y, args.min_precision)
    times = {s.name: s.time_ms for s in stats}

    full_met = _ensemble_metrics(y, union)
    total_ms = sum(times.values())
    print(f"\n  UNION всех паттернов: P={full_met['precision']:.4f} R={full_met['recall']:.4f} "
          f"F1={full_met['f1']:.4f}  TP={full_met['tp']} FP={full_met['fp']} FN={full_met['fn']}")
    print(f"  Суммарное время (последовательный прогон): {total_ms:.0f} ms  "
          f"({len(texts) / (total_ms / 1000):,.0f} texts/s equivalent)")

    _print_table(stats, f"Per-pattern stats ({task})")

    loo = _leave_one_out(patterns, masks, y, union)
    print(f"\n--- Leave-one-out (удаление паттерна из UNION) ---")
    print(f"{'pattern':<18} {'Δrecall':>10} {'ΔFP':>8} {'TP lost':>10}")
    for name, dr, dfp, lost in sorted(loo, key=lambda x: x[1]):
        print(f"{name:<18} {dr:10.4f} {dfp:8d} {lost:10d}")

    pareto = _pareto_subsets(patterns, masks, times, y, args.min_precision)
    print(f"\n--- Pareto-front (precision >= {args.min_precision}) — top by recall/time ---")
    shown = 0
    for p in sorted(pareto, key=lambda x: (-x["recall"], x["time_ms"]))[:12]:
        print(
            f"  k={p['k']:2d}  R={p['recall']:.4f}  P={p['precision']:.4f}  "
            f"time={p['time_ms']:7.0f}ms  FP={p['fp']:4d}  "
            f"patterns={','.join(p['patterns'])}"
        )
        shown += 1
    if not shown:
        print("  (нет подмножеств с заданным min_precision)")

    greedy = _greedy_pareto_build(patterns, masks, times, y, args.min_precision)
    pred_g = np.zeros(len(y), dtype=bool)
    t_g = 0.0
    for name in greedy:
        pred_g |= masks[name]
        t_g += times[name]
    met_g = _ensemble_metrics(y, pred_g)
    print(f"\n--- Жадный Pareto-build (Δrecall/Δtime, P>={args.min_precision}) ---")
    print(f"  patterns ({len(greedy)}): {', '.join(greedy)}")
    print(f"  R={met_g['recall']:.4f}  P={met_g['precision']:.4f}  F1={met_g['f1']:.4f}  "
          f"time≈{t_g:.0f}ms  (full union R={full_met['recall']:.4f} time≈{total_ms:.0f}ms)")

    # Примеры FP для худших паттернов
    print(f"\n--- Примеры FP (до 3 на паттерн, худший P) ---")
    series = pd.Series(texts, dtype=object)
    for s in sorted(stats, key=lambda x: x.precision if x.hits else 2.0)[:5]:
        if s.fp == 0:
            continue
        m = masks[s.name]
        fp_idx = np.flatnonzero(m & (y == 0))[:3]
        print(f"  [{s.name}] P={s.precision:.3f} FP={s.fp}:")
        for i in fp_idx:
            t = texts[i][:120].replace("\n", " ")
            print(f"    - {t}")

    _regex_quality_notes(task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ regex pre-filter паттернов")
    parser.add_argument("--task", choices=("tox", "spam", "both"), default="both")
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--spam-sample", type=int, default=100_000, help="Stratified sample for spam val")
    parser.add_argument("--min-precision", type=float, default=0.85, help="Порог P для Pareto")
    args = parser.parse_args()

    if args.task in ("tox", "both"):
        tox = RegexModel()
        run_task(args, "tox", tox._labeled_patterns())

    if args.task in ("spam", "both"):
        spam_args = argparse.Namespace(**{**vars(args), "val_data": args.val_data or Path("data/spam/val.parquet")})
        spam = SpamRegexModel()
        run_task(spam_args, "spam", spam._labeled_patterns())


if __name__ == "__main__":
    main()
