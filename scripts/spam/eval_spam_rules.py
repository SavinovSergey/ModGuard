"""Оценка правил спама на валидационных данных.

Правила: CAPS_WORD_DOUBLE_EXCL (слово капсом + !!), CAPS_WORD_6 (слово 7+ букв капсом).
Сработало → предсказание «спам». Основные метрики: доля ошибок от всех записей —
  FP% = FP / n * 100 (пометили не-спам как спам),
  FN% = FN / n * 100 (пропустили спам).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.features.spam_features import (
    matches_caps_word_double_excl_rule,
    matches_caps_word_rule,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precision/Recall правил спама (CAPS_WORD_DOUBLE_EXCL, CAPS_WORD_6) на val.parquet"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="spam_data/val.parquet",
        help="Parquet с колонками text, label",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Макс. число примеров (для быстрого прогона)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV с метриками по каждому правилу (опционально)",
    )
    parser.add_argument(
        "--fp-caps6-output",
        type=str,
        default=None,
        help="CSV со строками FP для правила CAPS_WORD_6 (не-спам, помеченный как спам). По умолчанию models/spam/tfidf/caps6_fp.csv",
    )
    args = parser.parse_args()

    print("Загрузка валидации...")
    df = pd.read_parquet(args.val_data, columns=["text", "label"])
    if args.max_samples is not None:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
        print(f"Подвыборка: {len(df)} примеров")
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].values
    if y_true.dtype == bool or np.issubdtype(y_true.dtype, np.bool_):
        y_true = y_true.astype(np.int64)
    else:
        y_true = (np.asarray(y_true) != 0).astype(np.int64)

    n = len(texts)
    n_spam = int(y_true.sum())
    print(f"Примеров: {n}, спам: {n_spam} ({100 * n_spam / n:.1f}%)\n")

    rules = [
        ("CAPS_WORD_DOUBLE_EXCL", "Слово капсом + !!", matches_caps_word_double_excl_rule),
        ("CAPS_WORD_6", "Слово из 7+ букв капсом", matches_caps_word_rule),
    ]

    rows = []
    for rule_id, rule_desc, rule_fn in rules:
        y_pred = np.array([1 if rule_fn(t) else 0 for t in texts], dtype=np.int64)
        n_positive = int(y_pred.sum())
        n_fp = int(((y_pred == 1) & (y_true == 0)).sum())
        n_fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp_pct = 100 * n_fp / n
        fn_pct = 100 * n_fn / n
        if n_positive == 0:
            prec = rec = f1 = 0.0
        else:
            prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        pct_triggered = 100 * n_positive / n
        rows.append({
            "rule_id": rule_id,
            "description": rule_desc,
            "n_triggered": n_positive,
            "pct_triggered": pct_triggered,
            "n_fp": n_fp,
            "fp_pct": fp_pct,
            "n_fn": n_fn,
            "fn_pct": fn_pct,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })
        print(f"Правило: {rule_id}")
        print(f"  Описание: {rule_desc}")
        print(f"  Срабатываний: {n_positive} ({pct_triggered:.1f}% от записей)")
        print(f"  FP: {n_fp} ({fp_pct:.2f}% от всех записей) — не-спам помечен как спам")
        print(f"  FN: {n_fn} ({fn_pct:.2f}% от всех записей) — спам не помечен")
        print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}\n")

        # Сохранение строк с FP для CAPS_WORD_6 (важно минимизировать FP)
        if rule_id == "CAPS_WORD_6" and n_fp > 0:
            fp_path = args.fp_caps6_output or "models/spam/tfidf/caps6_fp.csv"
            fp_path = Path(fp_path)
            fp_path.parent.mkdir(parents=True, exist_ok=True)
            fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
            fp_df = pd.DataFrame({
                "text": [texts[i] for i in fp_indices],
                "label": 0,
                "rule_id": rule_id,
            })
            fp_df.to_csv(fp_path, index=False, encoding="utf-8")
            print(f"  FP для CAPS_WORD_6 сохранены: {fp_path} ({len(fp_df)} строк)\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
        print(f"Метрики сохранены: {out_path}")


if __name__ == "__main__":
    main()
