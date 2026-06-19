#!/usr/bin/env python3
"""Ablation: char_wb transform time vs ngram_range and max_features."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.preprocessing.spam_processor import SpamTextProcessor


def bench_transform(vectorizer: TfidfVectorizer, texts: list[str], repeats: int = 3) -> tuple[float, float, int]:
    vectorizer.transform(texts[:1000])
    t0 = time.perf_counter()
    for _ in range(repeats):
        x = vectorizer.transform(texts)
    transform_ms = (time.perf_counter() - t0) / repeats * 1000
    nnz = x.nnz / len(texts)
    return transform_ms, nnz, x.shape[1]


def fit_and_bench(
    train_proc: list[str],
    val_proc: list[str],
    *,
    ngram_range: tuple[int, int],
    max_features: int,
    base: dict,
) -> tuple[float, float, float, int]:
    params = {**base, "ngram_range": ngram_range, "max_features": max_features}
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(**params)
    vectorizer.fit(train_proc)
    fit_ms = (time.perf_counter() - t0) * 1000
    transform_ms, nnz, vocab = bench_transform(vectorizer, val_proc)
    return fit_ms, transform_ms, nnz, vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-size", type=int, default=50_000)
    parser.add_argument("--val-path", default="data/toxicity/val.parquet")
    parser.add_argument("--train-path", default="data/spam/train.parquet")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    prep = SpamTextProcessor()

    train_df = pd.read_parquet(args.train_path, columns=["text"])
    if len(train_df) > args.fit_size:
        idx = rng.choice(len(train_df), size=args.fit_size, replace=False)
        train_df = train_df.iloc[idx]
    train_proc = prep.process_batch(train_df["text"].fillna("").astype(str).tolist())

    val_proc = prep.process_batch(
        pd.read_parquet(args.val_path)["text"].fillna("").astype(str).tolist()
    )

    base = dict(
        analyzer="char_wb",
        min_df=20,
        max_df=0.36240745617697456,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )

    print(f"Fit corpus: {len(train_proc):,} texts from {args.train_path}")
    print(f"Inference corpus: {len(val_proc):,} texts from {args.val_path}")
    print(f"Params: {base}\n")

    ngrams = [(1, 4), (2, 4), (2, 5), (3, 4), (3, 5)]
    max_feat_grid = [10_000, 20_000, 30_000, 45_000, 60_000, 90_000]

    print("=" * 90)
    print("Part 1: ngram_range sweep (max_features=30000)")
    print(f"{'ngram':>10} {'fit_ms':>10} {'transform_ms':>14} {'nnz/doc':>10} {'vocab':>10}")
    print("-" * 90)
    for ng in ngrams:
        fit_ms, tr_ms, nnz, vocab = fit_and_bench(
            train_proc, val_proc, ngram_range=ng, max_features=30_000, base=base
        )
        print(f"{str(ng):>10} {fit_ms:10.0f} {tr_ms:14.0f} {nnz:10.1f} {vocab:10d}")

    print("\n" + "=" * 90)
    print("Part 2: max_features sweep at fixed ngram_range")
    for ng in ngrams:
        print(f"\n--- ngram_range={ng} ---")
        print(f"{'max_features':>12} {'fit_ms':>10} {'transform_ms':>14} {'nnz/doc':>10} {'vocab':>10}")
        print("-" * 70)
        for mf in max_feat_grid:
            fit_ms, tr_ms, nnz, vocab = fit_and_bench(
                train_proc, val_proc, ngram_range=ng, max_features=mf, base=base
            )
            print(f"{mf:12d} {fit_ms:10.0f} {tr_ms:14.0f} {nnz:10.1f} {vocab:10d}")


if __name__ == "__main__":
    main()
