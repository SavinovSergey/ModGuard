"""Батчевый regex pre-filter: оптимизированный loop + опциональный ProcessPool."""
from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar

import numpy as np

PatternLabel = Tuple[str, re.Pattern]
T = TypeVar("T")

_DEFAULT_MIN_TEXTS_FOR_BUCKETS = 2_000
_DEFAULT_LENGTH_BUCKETS = 4
_DEFAULT_MIN_TEXTS_FOR_POOL = 10_000


def _length_bucket_indices(lengths: np.ndarray, n_buckets: int) -> List[np.ndarray]:
    """Индексы текстов с близкими длинами (равные по числу бакеты после сортировки)."""
    n = len(lengths)
    if n == 0:
        return []
    n_buckets = min(n_buckets, n)
    order = np.argsort(lengths, kind="stable")
    return [chunk for chunk in np.array_split(order, n_buckets) if len(chunk) > 0]


def _loop_classify_slice(
    texts: Sequence[str],
    patterns: Sequence[PatternLabel],
    empty_result: Callable[[], T],
    hit_result: Callable[[Dict[str, float]], T],
) -> List[T]:
    """Один проход по текстам; strip один раз; dict категорий только при hit."""
    empty = empty_result()
    pat_items = list(patterns)
    results: List[T] = []
    for text in texts:
        if not text:
            results.append(empty)
            continue
        s = text.strip()
        if not s:
            results.append(empty)
            continue
        matched: Dict[str, float] | None = None
        for name, pattern in pat_items:
            if pattern.search(s):
                if matched is None:
                    matched = {}
                matched[name] = 1.0
        results.append(hit_result(matched) if matched else empty)
    return results


def _pool_worker(
    payload: Tuple[List[str], List[Tuple[str, str, int]], str],
) -> List[Dict[str, Any]]:
    texts, pattern_specs, model_kind = payload
    patterns: List[PatternLabel] = [
        (name, re.compile(pat, flags=flags)) for name, pat, flags in pattern_specs
    ]
    if model_kind == "tox":
        from app.models.toxicity.regex_model import RegexModel

        empty = RegexModel.empty_result
        hit = RegexModel._hit_result
    else:
        from app.models.spam.regex_model import SpamRegexModel

        empty = SpamRegexModel._empty
        hit = SpamRegexModel._hit_result

    return _loop_classify_slice(texts, patterns, empty, hit)


def batch_regex_classify(
    texts: List[str],
    patterns: Sequence[PatternLabel],
    *,
    empty_result: Callable[[], T],
    hit_result: Callable[[Dict[str, float]], T],
    length_buckets: int = _DEFAULT_LENGTH_BUCKETS,
    min_texts_for_buckets: int = _DEFAULT_MIN_TEXTS_FOR_BUCKETS,
    pool_workers: int | None = None,
    min_texts_for_pool: int = _DEFAULT_MIN_TEXTS_FOR_POOL,
    pool_tag: str | None = None,
) -> List[T]:
    """
    Классификация батча regex-паттернами.

    Тексты переменной длины не склеиваются. Для крупных батчей индексы
    группируются по длине (локальность кэша). ProcessPool включается только
    явно через REGEX_BATCH_WORKERS>0 и n >= min_texts_for_pool (offline/validate).
    """
    n = len(texts)
    if n == 0:
        return []

    workers = pool_workers
    if workers is None:
        workers = int(os.environ.get("REGEX_BATCH_WORKERS", "0"))

    if workers > 1 and n >= min_texts_for_pool and pool_tag is not None:
        pattern_specs = [(name, p.pattern, p.flags) for name, p in patterns]
        chunks = [texts[i:j] for i, j in _chunk_ranges(n, workers)]
        payload = [(chunk, pattern_specs, pool_tag) for chunk in chunks if chunk]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            parts = executor.map(_pool_worker, payload)
        out: List[T] = []
        for part in parts:
            out.extend(part)  # type: ignore[arg-type]
        return out

    results: List[T | None] = [None] * n
    lengths = np.fromiter((len(t) if t else 0 for t in texts), dtype=np.int32, count=n)

    use_buckets = length_buckets > 1 and n >= min_texts_for_buckets
    index_groups = (
        _length_bucket_indices(lengths, length_buckets)
        if use_buckets
        else [np.arange(n, dtype=np.intp)]
    )

    for indices in index_groups:
        sub_texts = [texts[i] for i in indices]
        sub_results = _loop_classify_slice(sub_texts, patterns, empty_result, hit_result)
        for j, orig_i in enumerate(indices):
            results[orig_i] = sub_results[j]

    return results  # type: ignore[return-value]


def _chunk_ranges(n: int, workers: int) -> List[Tuple[int, int]]:
    step = (n + workers - 1) // workers
    return [(i, min(i + step, n)) for i in range(0, n, step)]
