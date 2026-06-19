"""Анализ tfidf_transform и extra_features: время и связь с таргетом."""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.features.spam_features import (
    SPAM_FEATURE_NAMES_MINIMAL,
    _repeated_chars_ratio,
    _single_letter_token_ratio,
    _space_ratio,
    _typo_score,
    extract_spam_features_batch,
)
from app.models.spam.tfidf_model import SpamTfidfModel
from app.preprocessing.spam_processor import SpamTextProcessor

_FEATURE_FNS = {
    "length_chars_log1p": lambda t: float(np.log1p(len(str(t).strip()))),
    "repeated_chars_ratio": lambda t: float(_repeated_chars_ratio(str(t).strip())),
    "typo_score": lambda t: float(_typo_score(str(t).strip())),
    "space_ratio": lambda t: float(_space_ratio(str(t).strip())),
    "single_letter_token_ratio": lambda t: float(_single_letter_token_ratio(str(t).strip())),
}


def _load_sample(path: Path, n: int, seed: int = 42) -> tuple[list[str], np.ndarray]:
    df = pd.read_parquet(path)
    y = df["label"].astype(int).to_numpy()
    texts = df["text"].fillna("").astype(str).tolist()
    if n >= len(texts):
        return texts, y
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = min(len(pos), max(1, n // 10))
    n_neg = n - n_pos
    pick = np.concatenate([
        rng.choice(pos, n_pos, replace=False),
        rng.choice(neg, n_neg, replace=False),
    ])
    pick.sort()
    return [texts[i] for i in pick], y[pick]


def _bench(fn, repeat: int = 3) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat * 1000.0


def analyze_extra_features(texts: list[str], y: np.ndarray, model_dir: Path) -> pd.DataFrame:
    n = len(texts)
    rows = []

    # timing per feature
    for name, fn in _FEATURE_FNS.items():
        ms = _bench(lambda f=fn: [f(t) for t in texts])
        rows.append({"component": name, "kind": "extra_feature", "time_ms": ms})

    # batch baseline
    ms_batch = _bench(lambda: extract_spam_features_batch(texts, SPAM_FEATURE_NAMES_MINIMAL))
    rows.append({"component": "extract_batch_total", "kind": "extra_feature", "time_ms": ms_batch})

    # values matrix
    X = extract_spam_features_batch(texts, SPAM_FEATURE_NAMES_MINIMAL)

    # model coef for manual features (last 5 cols)
    coef_manual = None
    model_path = model_dir / "model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        if hasattr(clf, "coef_"):
            coef_manual = clf.coef_.ravel()[-5:]

    target_rows = []
    for j, name in enumerate(SPAM_FEATURE_NAMES_MINIMAL):
        col = X[:, j]
        r, p = stats.pearsonr(col, y)
        spam_mean = col[y == 1].mean()
        ham_mean = col[y == 0].mean()
        # single-feature AUC via rank
        auc = stats.mannwhitneyu(col[y == 1], col[y == 0], alternative="two-sided")
        # proper AUC
        from sklearn.metrics import roc_auc_score

        try:
            auc_val = roc_auc_score(y, col)
            if auc_val < 0.5:
                auc_val = 1.0 - auc_val
        except ValueError:
            auc_val = float("nan")
        coef = float(coef_manual[j]) if coef_manual is not None else float("nan")
        target_rows.append({
            "feature": name,
            "pearson_r": r,
            "p_value": p,
            "mean_spam": spam_mean,
            "mean_ham": ham_mean,
            "delta_mean": spam_mean - ham_mean,
            "auc": auc_val,
            "model_coef": coef,
            "time_ms": next(r["time_ms"] for r in rows if r["component"] == name),
            "time_pct": 0.0,
        })

    tr = pd.DataFrame(target_rows)
    total_t = tr["time_ms"].sum()
    tr["time_pct"] = 100.0 * tr["time_ms"] / total_t if total_t else 0.0
    tr["batch_overhead_ms"] = ms_batch - total_t
    return tr, ms_batch


def analyze_tfidf_stages(texts: list[str], y: np.ndarray, model_dir: Path) -> dict:
    model = SpamTfidfModel()
    model.load(
        model_path=str(model_dir / "model.pkl"),
        vectorizer_path=str(model_dir / "vectorizer.pkl"),
        params_path=str(model_dir / "params.json"),
    )
    prep = SpamTextProcessor()

    processed = prep.process_batch(texts)
    non_empty_idx = [i for i, p in enumerate(processed) if p and p.strip()]
    proc_texts = [processed[i] for i in non_empty_idx]
    raw_texts = [texts[i] for i in non_empty_idx]
    y_ne = y[non_empty_idx]

    ms_prep = _bench(lambda: prep.process_batch(texts))
    ms_tfidf = _bench(lambda: model.vectorizer.transform(proc_texts))
    ms_extra = _bench(lambda: extract_spam_features_batch(raw_texts, SPAM_FEATURE_NAMES_MINIMAL))
    ms_hstack = 0.0
    ms_predict = 0.0

    from scipy.sparse import csr_matrix, hstack as sparse_hstack

    X_tfidf = model.vectorizer.transform(proc_texts)
    X_feat = extract_spam_features_batch(raw_texts, SPAM_FEATURE_NAMES_MINIMAL)
    X_feat = model.scaler.transform(X_feat)
    ms_hstack = _bench(
        lambda: sparse_hstack([X_tfidf, csr_matrix(X_feat.astype(np.float64))])
    )
    X = sparse_hstack([X_tfidf, csr_matrix(X_feat.astype(np.float64))])
    ms_predict = _bench(lambda: model.model.predict_proba(X))

    # TF-IDF target link: top terms by coef, hit rate on spam vs ham
    coef = model.model.coef_.ravel()
    n_tfidf = X_tfidf.shape[1]
    tfidf_coef = coef[:n_tfidf]
    vocab = np.array(model.vectorizer.get_feature_names_out())
    top_pos_idx = np.argsort(tfidf_coef)[-15:][::-1]
    top_neg_idx = np.argsort(tfidf_coef)[:15]

    term_rows = []
    X_bin = (X_tfidf > 0).astype(np.int8)
    for idx in top_pos_idx:
        hits = np.asarray(X_bin[:, idx].todense()).ravel()
        term_rows.append({
            "term": vocab[idx],
            "coef": tfidf_coef[idx],
            "hit_rate_spam": hits[y_ne == 1].mean() if (y_ne == 1).any() else 0,
            "hit_rate_ham": hits[y_ne == 0].mean() if (y_ne == 0).any() else 0,
        })
    terms_df = pd.DataFrame(term_rows)

    return {
        "ms_prep": ms_prep,
        "ms_tfidf": ms_tfidf,
        "ms_extra": ms_extra,
        "ms_hstack": ms_hstack,
        "ms_predict": ms_predict,
        "ms_build_X_total": ms_tfidf + ms_extra + ms_hstack,
        "n_non_empty": len(non_empty_idx),
        "top_spam_terms": terms_df,
        "n_tfidf_features": n_tfidf,
        "n_manual_features": len(SPAM_FEATURE_NAMES_MINIMAL),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-data", default="data/spam/val.parquet")
    parser.add_argument("--sample", type=int, default=100_000)
    parser.add_argument("--model-dir", default="models/spam/tfidf")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    texts, y = _load_sample(Path(args.val_data), args.sample)
    print(f"Данные: {args.val_data}  n={len(texts)}  spam={y.sum()} ({100*y.mean():.2f}%)")

    print("\n" + "=" * 72)
    print("EXTRA FEATURES — время и связь с таргетом")
    print("=" * 72)
    feat_df, ms_batch = analyze_extra_features(texts, y, model_dir)
    print(f"\n{'feature':<28} {'ms':>8} {'%time':>6} {'r':>7} {'AUC':>6} {'Δmean':>8} {'coef':>8}")
    print("-" * 72)
    for _, row in feat_df.iterrows():
        print(
            f"{row['feature']:<28} {row['time_ms']:8.1f} {row['time_pct']:5.1f}% "
            f"{row['pearson_r']:7.4f} {row['auc']:6.3f} {row['delta_mean']:8.4f} {row['model_coef']:8.3f}"
        )
    overhead = ms_batch - feat_df["time_ms"].sum()
    print(f"{'batch_overhead (loop)':<28} {overhead:8.1f}")
    print(f"{'extract_batch_total':<28} {ms_batch:8.1f}")

    print("\n" + "=" * 72)
    print("TF-IDF PIPELINE — разбивка времени (_build_X path)")
    print("=" * 72)
    tf = analyze_tfidf_stages(texts, y, model_dir)
    total = tf["ms_tfidf"] + tf["ms_extra"] + tf["ms_hstack"]
    print(f"  preprocess (SpamTextProcessor): {tf['ms_prep']:.0f} ms")
    print(f"  tfidf_transform:              {tf['ms_tfidf']:.0f} ms  ({100*tf['ms_tfidf']/total:.0f}% ML-mat)")
    print(f"  extra_features + scaler*:     {tf['ms_extra']:.0f} ms  ({100*tf['ms_extra']/total:.0f}%)  *scaler inside batch")
    print(f"  hstack:                       {tf['ms_hstack']:.0f} ms  ({100*tf['ms_hstack']/total:.0f}%)")
    print(f"  predict_proba:                {tf['ms_predict']:.0f} ms")
    print(f"  non_empty texts:              {tf['n_non_empty']} / {len(texts)}")
    print(f"  tfidf dims: {tf['n_tfidf_features']} + manual {tf['n_manual_features']}")

    print("\n  Top-15 TF-IDF char-terms по coef (pro-spam):")
    print(f"  {'term':<12} {'coef':>8} {'hit_spam':>9} {'hit_ham':>9} {'lift':>6}")
    for _, row in tf["top_spam_terms"].iterrows():
        lift = row["hit_rate_spam"] / row["hit_rate_ham"] if row["hit_rate_ham"] > 0 else float("inf")
        term = repr(row["term"])[:12]
        print(
            f"  {term:<12} {row['coef']:8.2f} {row['hit_rate_spam']:9.4f} "
            f"{row['hit_rate_ham']:9.4f} {lift:6.1f}x"
        )


if __name__ == "__main__":
    main()
