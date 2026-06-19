"""Профилирование этапов classify spam/tox (без API, PG, Redis, MQ).

Разбивает пайплайн на этапы и измеряет вклад каждого в общее время
на тех же батчах, что validate_toxicity / validate_spam.

Пример:
  python scripts/run/profile_classify_stages.py \\
    --val-data data/toxicity/val.parquet \\
    --batch-size 1000 --warmup 1 --repeat 3
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.features.spam_features import (  # noqa: E402
    matches_caps_word_double_excl_rule,
    matches_caps_word_rule,
)
from app.models.spam.tfidf_model import SpamTfidfModel  # noqa: E402
from app.models.toxicity.tfidf_model import TfidfModel  # noqa: E402
from app.services.spam.service import SpamService  # noqa: E402
from app.services.toxicity.service import ToxicityService  # noqa: E402
from app.loader import get_spam_regex_model  # noqa: E402
from app.models.toxicity.regex_model import RegexModel  # noqa: E402
from app.core.model_manager import ModelManager  # noqa: E402


@dataclass
class StageAccumulator:
    name: str
    total_s: float = 0.0
    items: int = 0

    def add(self, elapsed_s: float, n: int) -> None:
        self.total_s += elapsed_s
        self.items += n

    @property
    def total_ms(self) -> float:
        return self.total_s * 1000.0

    @property
    def ms_per_item(self) -> float:
        return self.total_ms / self.items if self.items else 0.0


@dataclass
class PipelineProfile:
    pipeline: str
    stages: Dict[str, StageAccumulator] = field(default_factory=dict)
    total_items: int = 0

    def stage(self, name: str) -> StageAccumulator:
        if name not in self.stages:
            self.stages[name] = StageAccumulator(name=name)
        return self.stages[name]

    def add_total(self, elapsed_s: float, n: int) -> None:
        self.stage("total").add(elapsed_s, n)

    def merge(self, other: "PipelineProfile") -> None:
        self.total_items += other.total_items
        for name, acc in other.stages.items():
            s = self.stage(name)
            s.total_s += acc.total_s
            s.items += acc.items


def _load_texts(path: Path, text_col: str) -> List[str]:
    df = pd.read_parquet(path)
    if text_col not in df.columns:
        raise ValueError(f"Колонка {text_col!r} не найдена в {path}")
    return [str(t) if t is not None else "" for t in df[text_col].tolist()]


def _time_call(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def profile_toxicity_validate_like(texts: List[str], tox_model: TfidfModel) -> PipelineProfile:
    """TfidfModel.predict_batch без regex (как validate_toxicity)."""
    prof = PipelineProfile(pipeline="toxicity_validate")
    n = len(texts)
    prof.total_items = n

    t0 = time.perf_counter()
    processed = tox_model.preprocess_batch(texts)
    prof.stage("2_preprocess").add(time.perf_counter() - t0, n)

    non_empty = [t for t in processed if t and t.strip()]
    if not non_empty:
        return prof

    t0 = time.perf_counter()
    X = tox_model.vectorizer.transform(non_empty)
    prof.stage("3_vectorize").add(time.perf_counter() - t0, len(non_empty))

    t0 = time.perf_counter()
    _ = tox_model.model.predict_proba(X)
    prof.stage("4_predict_proba").add(time.perf_counter() - t0, len(non_empty))

    ml_ms = sum(prof.stages[k].total_ms for k in prof.stages if k != "1_regex_prefilter")
    prof.stage("ml_subtotal").add(ml_ms / 1000.0, n)
    return prof


def profile_spam_validate_like(texts: List[str], spam_model: SpamTfidfModel) -> PipelineProfile:
    """SpamTfidfModel.predict_batch без regex-сервиса (как validate_spam)."""
    prof = PipelineProfile(pipeline="spam_validate")
    n = len(texts)
    prof.total_items = n

    t0 = time.perf_counter()
    rule_hits = [
        matches_caps_word_double_excl_rule(t) or matches_caps_word_rule(t) for t in texts
    ]
    prof.stage("2_caps_rules").add(time.perf_counter() - t0, n)

    tfidf_texts = [t for t, hit in zip(texts, rule_hits) if not hit]
    if not tfidf_texts:
        return prof

    t0 = time.perf_counter()
    processed = spam_model.preprocessor.process_batch(tfidf_texts)
    prof.stage("3_preprocess").add(time.perf_counter() - t0, len(tfidf_texts))

    non_empty_processed: List[str] = []
    non_empty_raw: List[str] = []
    for raw, p in zip(tfidf_texts, processed):
        if p and p.strip():
            non_empty_processed.append(p)
            non_empty_raw.append(raw)

    if not non_empty_processed:
        return prof

    t0 = time.perf_counter()
    if spam_model.use_caps_rest_split and spam_model.vectorizer_caps is not None:
        from app.preprocessing.spam_processor import split_caps_rest_batch
        from scipy.sparse import hstack as sparse_hstack

        caps_parts, rest_parts = split_caps_rest_batch(non_empty_processed)
        X_caps = spam_model.vectorizer_caps.transform(caps_parts)
        X_rest = spam_model.vectorizer.transform(rest_parts)
        X_tfidf = sparse_hstack([X_caps, X_rest])
    else:
        X_tfidf = spam_model.vectorizer.transform(non_empty_processed)
    prof.stage("4_tfidf_transform").add(time.perf_counter() - t0, len(non_empty_processed))

    X = X_tfidf
    if spam_model.use_extra_features and spam_model.scaler is not None:
        from app.features.spam_features import extract_spam_features_batch
        from scipy.sparse import csr_matrix, hstack as sparse_hstack

        t0 = time.perf_counter()
        X_feat = extract_spam_features_batch(non_empty_raw, feature_names=spam_model.spam_feature_names)
        prof.stage("5_extra_features").add(time.perf_counter() - t0, len(non_empty_raw))

        t0 = time.perf_counter()
        X_feat = spam_model.scaler.transform(X_feat)
        X_feat_sparse = csr_matrix(X_feat.astype("float64"))
        X = sparse_hstack([X_tfidf, X_feat_sparse])
        prof.stage("6_hstack").add(time.perf_counter() - t0, len(non_empty_raw))

    t0 = time.perf_counter()
    _ = spam_model.model.predict_proba(X)
    prof.stage("7_predict_proba").add(time.perf_counter() - t0, len(non_empty_processed))

    ml_ms = sum(prof.stages[k].total_ms for k in prof.stages if k.startswith(("2_", "3_", "4_", "5_", "6_", "7_")))
    prof.stage("ml_subtotal").add(ml_ms / 1000.0, n)
    return prof


def profile_toxicity_batch(
    texts: List[str],
    tox_model: TfidfModel,
    regex_model,
) -> PipelineProfile:
    """Этапы ToxicityService + TfidfModel (regex → preprocess → vectorize → predict)."""
    prof = PipelineProfile(pipeline="toxicity")
    n = len(texts)
    prof.total_items = n

    # --- regex pre-filter (как в ToxicityService) ---
    miss_indices: List[int] = []
    miss_texts: List[str] = []
    for i, text in enumerate(texts):
        if text:
            miss_indices.append(i)
            miss_texts.append(text)

    t0 = time.perf_counter()
    regex_results = regex_model.predict_batch(miss_texts) if regex_model.is_loaded else [None] * len(miss_texts)
    prof.stage("1_regex_prefilter").add(time.perf_counter() - t0, len(miss_texts))

    ml_texts: List[str] = []
    for j, text in enumerate(miss_texts):
        r = regex_results[j] if regex_results and j < len(regex_results) and regex_results[j] is not None else {}
        if not r.get("is_toxic"):
            ml_texts.append(text)

    if not ml_texts:
        prof.add_total(0.0, n)
        return prof

    # --- preprocess ---
    t0 = time.perf_counter()
    processed = tox_model.preprocess_batch(ml_texts)
    prof.stage("2_preprocess").add(time.perf_counter() - t0, len(ml_texts))

    non_empty = [t for t in processed if t and t.strip()]
    if not non_empty:
        prof.add_total(0.0, n)
        return prof

    # --- vectorize ---
    t0 = time.perf_counter()
    X = tox_model.vectorizer.transform(non_empty)
    prof.stage("3_vectorize").add(time.perf_counter() - t0, len(non_empty))

    # --- predict ---
    t0 = time.perf_counter()
    _ = tox_model.model.predict_proba(X)
    prof.stage("4_predict_proba").add(time.perf_counter() - t0, len(non_empty))

    ml_ms = sum(
        prof.stages[k].total_ms
        for k in ("2_preprocess", "3_vectorize", "4_predict_proba")
        if k in prof.stages
    )
    prof.stage("ml_subtotal").add(ml_ms / 1000.0, len(ml_texts))

    return prof


def profile_spam_batch(
    texts: List[str],
    spam_model: SpamTfidfModel,
    regex_model,
) -> PipelineProfile:
    """Этапы SpamService + SpamTfidfModel."""
    prof = PipelineProfile(pipeline="spam")
    n = len(texts)
    prof.total_items = n

    # --- regex pre-filter (SpamService phase 1) ---
    t0 = time.perf_counter()
    regex_results = regex_model.predict_batch(texts) if regex_model.is_loaded else [None] * n
    prof.stage("1_regex_prefilter").add(time.perf_counter() - t0, n)

    ml_texts: List[str] = []
    for i, text in enumerate(texts):
        if not text:
            continue
        r = regex_results[i] if regex_results[i] is not None else {}
        if not r.get("is_spam"):
            ml_texts.append(text)

    if not ml_texts:
        return prof

    # --- caps rules (внутри SpamTfidfModel.predict_batch) ---
    t0 = time.perf_counter()
    rule_hits = [
        matches_caps_word_double_excl_rule(t) or matches_caps_word_rule(t) for t in ml_texts
    ]
    prof.stage("2_caps_rules").add(time.perf_counter() - t0, len(ml_texts))

    tfidf_texts = [t for t, hit in zip(ml_texts, rule_hits) if not hit]
    if not tfidf_texts:
        return prof

    # --- preprocess ---
    t0 = time.perf_counter()
    processed = spam_model.preprocessor.process_batch(tfidf_texts)
    prof.stage("3_preprocess").add(time.perf_counter() - t0, len(tfidf_texts))

    non_empty_processed: List[str] = []
    non_empty_raw: List[str] = []
    for raw, p in zip(tfidf_texts, processed):
        if p and p.strip():
            non_empty_processed.append(p)
            non_empty_raw.append(raw)

    if not non_empty_processed:
        return prof

    # --- tfidf transform ---
    t0 = time.perf_counter()
    if spam_model.use_caps_rest_split and spam_model.vectorizer_caps is not None:
        from app.preprocessing.spam_processor import split_caps_rest_batch

        caps_parts, rest_parts = split_caps_rest_batch(non_empty_processed)
        X_caps = spam_model.vectorizer_caps.transform(caps_parts)
        X_rest = spam_model.vectorizer.transform(rest_parts)
        from scipy.sparse import hstack as sparse_hstack

        X_tfidf = sparse_hstack([X_caps, X_rest])
    else:
        X_tfidf = spam_model.vectorizer.transform(non_empty_processed)
    prof.stage("4_tfidf_transform").add(time.perf_counter() - t0, len(non_empty_processed))

    X = X_tfidf
    if spam_model.use_extra_features and spam_model.scaler is not None:
        from app.features.spam_features import extract_spam_features_batch
        from scipy.sparse import csr_matrix, hstack as sparse_hstack

        t0 = time.perf_counter()
        X_feat = extract_spam_features_batch(non_empty_raw, feature_names=spam_model.spam_feature_names)
        prof.stage("5_extra_features").add(time.perf_counter() - t0, len(non_empty_raw))

        t0 = time.perf_counter()
        X_feat = spam_model.scaler.transform(X_feat)
        X_feat_sparse = csr_matrix(X_feat.astype("float64"))
        X = sparse_hstack([X_tfidf, X_feat_sparse])
        prof.stage("6_hstack").add(time.perf_counter() - t0, len(non_empty_raw))

    # --- predict ---
    t0 = time.perf_counter()
    _ = spam_model.model.predict_proba(X)
    prof.stage("7_predict_proba").add(time.perf_counter() - t0, len(non_empty_processed))

    ml_ms = sum(
        prof.stages[k].total_ms
        for k in prof.stages
        if k.startswith(("2_", "3_", "4_", "5_", "6_", "7_"))
    )
    prof.stage("ml_subtotal").add(ml_ms / 1000.0, len(ml_texts))

    return prof


def run_profile(
    texts: List[str],
    batch_size: int,
    tox_model: TfidfModel,
    spam_model: SpamTfidfModel,
    tox_regex,
    spam_regex,
    tox_service: ToxicityService,
    spam_service: SpamService,
) -> Tuple[
    PipelineProfile,
    PipelineProfile,
    PipelineProfile,
    PipelineProfile,
    PipelineProfile,
    PipelineProfile,
    PipelineProfile,
    PipelineProfile,
]:
    tox_prof = PipelineProfile(pipeline="toxicity (worker)")
    spam_prof = PipelineProfile(pipeline="spam (worker)")
    tox_val = PipelineProfile(pipeline="toxicity (validate)")
    spam_val = PipelineProfile(pipeline="spam (validate)")
    tox_e2e = PipelineProfile(pipeline="toxicity_e2e")
    spam_e2e = PipelineProfile(pipeline="spam_e2e")
    tox_pred = PipelineProfile(pipeline="tox_predict_batch")
    spam_pred = PipelineProfile(pipeline="spam_predict_batch")

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        n = len(batch)

        t0 = time.perf_counter()
        _ = tox_service.classify_batch(batch, preferred_model="tfidf")
        tox_e2e.add_total(time.perf_counter() - t0, n)

        t0 = time.perf_counter()
        _ = spam_service.classify_batch(batch)
        spam_e2e.add_total(time.perf_counter() - t0, n)

        t0 = time.perf_counter()
        _ = tox_model.predict_batch(batch)
        tox_pred.add_total(time.perf_counter() - t0, n)

        t0 = time.perf_counter()
        _ = spam_model.predict_batch(batch)
        spam_pred.add_total(time.perf_counter() - t0, n)

        tox_prof.merge(profile_toxicity_batch(batch, tox_model, tox_regex))
        spam_prof.merge(profile_spam_batch(batch, spam_model, spam_regex))
        tox_val.merge(profile_toxicity_validate_like(batch, tox_model))
        spam_val.merge(profile_spam_validate_like(batch, spam_model))

    return tox_prof, spam_prof, tox_val, spam_val, tox_e2e, spam_e2e, tox_pred, spam_pred


def _print_profile(
    prof: PipelineProfile,
    e2e: PipelineProfile | None = None,
    *,
    title: str | None = None,
    n_texts: int | None = None,
) -> None:
    base_ms = e2e.stage("total").total_ms if e2e else prof.stages.get("ml_subtotal", StageAccumulator("")).total_ms
    if base_ms <= 0:
        base_ms = sum(s.total_ms for s in prof.stages.values() if s.name not in ("total", "ml_subtotal"))

    header = title or prof.pipeline
    n = n_texts or prof.total_items
    print(f"\n=== {header} ({n} текстов) ===")
    if e2e:
        e2e_ms = e2e.stage("total").total_ms
        print(f"  e2e: {e2e_ms:,.1f} ms  ({n / (e2e_ms / 1000):,.0f} msg/s)")

    ordered = sorted(
        prof.stages.items(),
        key=lambda kv: (0 if kv[0] == "total" else 1, kv[0]),
    )
    for name, acc in ordered:
        if name in ("total", "ml_subtotal"):
            continue
        pct = 100.0 * acc.total_ms / base_ms if base_ms else 0.0
        print(
            f"  {acc.name:22s} {acc.total_ms:10,.1f} ms  "
            f"({acc.ms_per_item:6.3f} ms/item)  {pct:5.1f}%"
        )

    if "ml_subtotal" in prof.stages:
        acc = prof.stages["ml_subtotal"]
        pct = 100.0 * acc.total_ms / base_ms if base_ms else 0.0
        print(
            f"  {'ml_subtotal':22s} {acc.total_ms:10,.1f} ms  "
            f"({acc.ms_per_item:6.3f} ms/item)  {pct:5.1f}%  ← ML без regex"
        )

    regex_acc = prof.stages.get("1_regex_prefilter")
    if regex_acc:
        pct = 100.0 * regex_acc.total_ms / base_ms if base_ms else 0.0
        print(
            f"  {'regex (phase 1)':22s} {regex_acc.total_ms:10,.1f} ms  "
            f"({regex_acc.ms_per_item:6.3f} ms/item)  {pct:5.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Профилирование этапов classify spam/tox")
    parser.add_argument("--val-data", default="data/toxicity/val.parquet")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=1, help="Прогревовых проходов (не в отчёт)")
    parser.add_argument("--repeat", type=int, default=1, help="Проходов в отчёт (усреднение)")
    parser.add_argument("--tox-model-dir", default="models/toxicity/tfidf")
    parser.add_argument("--spam-model-dir", default="models/spam/tfidf")
    args = parser.parse_args()

    texts = _load_texts(Path(args.val_data), args.text_col)
    print(f"Загружено {len(texts)} текстов из {args.val_data}")

    tox_dir = Path(args.tox_model_dir)
    spam_dir = Path(args.spam_model_dir)

    tox_model = TfidfModel()
    tox_model.load(
        model_path=str(tox_dir / "model.pkl"),
        vectorizer_path=str(tox_dir / "vectorizer.pkl"),
        params_path=str(tox_dir / "params.json"),
    )

    spam_model = SpamTfidfModel()
    spam_model.load(
        model_path=str(spam_dir / "model.pkl"),
        vectorizer_path=str(spam_dir / "vectorizer.pkl"),
        params_path=str(spam_dir / "params.json"),
    )

    tox_regex = RegexModel()
    tox_regex.load()
    spam_regex = get_spam_regex_model()

    mm = ModelManager()
    mm.register_model("regex", tox_regex)
    mm.register_model("tfidf", tox_model)
    tox_service = ToxicityService(mm)
    spam_service = SpamService(spam_model=spam_model, spam_regex_model=spam_regex)

    print(
        f"spam: use_extra_features={spam_model.use_extra_features}, "
        f"features={spam_model.spam_feature_names}, "
        f"caps_rest_split={spam_model.use_caps_rest_split}"
    )
    print(
        f"tox: lemmatization={getattr(tox_model.text_processor, 'use_lemmatization', '?')}, "
        f"max_features={getattr(tox_model.vectorizer, 'max_features', '?')}"
    )
    print(f"batch_size={args.batch_size}, warmup={args.warmup}, repeat={args.repeat}")

    for _ in range(args.warmup):
        run_profile(
            texts, args.batch_size, tox_model, spam_model,
            tox_regex, spam_regex, tox_service, spam_service,
        )

    tox_sum = PipelineProfile("toxicity")
    spam_sum = PipelineProfile("spam")
    tox_val_sum = PipelineProfile("toxicity_validate")
    spam_val_sum = PipelineProfile("spam_validate")
    tox_e2e_sum = PipelineProfile("toxicity_e2e")
    spam_e2e_sum = PipelineProfile("spam_e2e")
    tox_pred_sum = PipelineProfile("tox_predict_batch")
    spam_pred_sum = PipelineProfile("spam_predict_batch")

    for r in range(args.repeat):
        (
            tox_p, spam_p, tox_val, spam_val,
            tox_e2e, spam_e2e, tox_pred, spam_pred,
        ) = run_profile(
            texts, args.batch_size, tox_model, spam_model,
            tox_regex, spam_regex, tox_service, spam_service,
        )
        for p in (tox_p, spam_p, tox_val, spam_val, tox_e2e, spam_e2e, tox_pred, spam_pred):
            for name, acc in p.stages.items():
                p.stages[name] = StageAccumulator(
                    name=name,
                    total_s=acc.total_s / args.repeat,
                    items=acc.items,
                )
        tox_sum.merge(tox_p)
        spam_sum.merge(spam_p)
        tox_val_sum.merge(tox_val)
        spam_val_sum.merge(spam_val)
        tox_e2e_sum.merge(tox_e2e)
        spam_e2e_sum.merge(spam_e2e)
        tox_pred_sum.merge(tox_pred)
        spam_pred_sum.merge(spam_pred)

    n_texts = len(texts)

    print("\n--- validate-скрипты: model.predict_batch (без regex-сервиса) ---")
    _print_profile(tox_val_sum, tox_pred_sum, title="toxicity validate-like", n_texts=n_texts)
    _print_profile(spam_val_sum, spam_pred_sum, title="spam validate-like", n_texts=n_texts)

    print("\n--- worker: SpamService / ToxicityService (regex + ML) ---")
    _print_profile(tox_sum, tox_e2e_sum, title="toxicity worker", n_texts=n_texts)
    _print_profile(spam_sum, spam_e2e_sum, title="spam worker", n_texts=n_texts)

    tox_ms = tox_pred_sum.stage("total").total_ms
    spam_ms = spam_pred_sum.stage("total").total_ms
    if tox_ms > 0:
        ratio = spam_ms / tox_ms
        print(f"\n=== Сравнение (validate-like predict_batch) ===")
        print(f"  spam / tox: {ratio:.2f}x  (spam {spam_ms:,.0f} ms vs tox {tox_ms:,.0f} ms)")

    tox_w = tox_e2e_sum.stage("total").total_ms
    spam_w = spam_e2e_sum.stage("total").total_ms
    if tox_w > 0:
        print(f"  worker classify_batch: spam / tox = {spam_w / tox_w:.2f}x")


if __name__ == "__main__":
    main()
