"""Валидация спам-модели на отложенных данных.

Режимы (--model-type):
  tfidf — TF-IDF + ручные правила: PR-кривая, порог (max F1 при precision >= 0.90),
          FP/FN с признаками, важность признаков, опционально обновление params.json.
  regex  — только regex pre-filter: Precision/Recall/F1 при фиксированном пороге,
          FP/FN с указанием сработавших категорий (earnings, cta_links, casino и т.д.).

Режим оценки на тесте: передайте `--eval-mode test`, тогда порог не подбирается заново,
а используется `model.optimal_threshold` (или `--threshold`, если задан).

Примеры:
  python scripts/spam/validate_spam.py --model-type tfidf --val-data spam_data/val.parquet
  python scripts/spam/validate_spam.py --model-type regex --val-data spam_data/val.parquet

Требуется: pip install matplotlib (для tfidf)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.shared.common import find_threshold_max_f1_min_precision

MIN_PRECISION = 0.90


def _run_tfidf(
    args, model_dir: Path, df: pd.DataFrame, texts: list, y_true: np.ndarray, batch_size: int
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.features.spam_features import SPAM_FEATURE_NAMES, extract_spam_features
    from app.models.spam.tfidf_model import SpamTfidfModel

    print(f"Загрузка TF-IDF модели из {model_dir}...")
    model = SpamTfidfModel()
    model.load(
        model_path=str(model_dir / "model.pkl"),
        vectorizer_path=str(model_dir / "vectorizer.pkl"),
    )
    print("Предсказания: полный пайплайн (TF-IDF + ручные признаки + правило caps+!!).")

    results = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Получение предсказаний tfidf...",
    ):
        batch = texts[start : start + batch_size]
        results.extend(model.predict_batch(batch))
    print()

    y_proba = np.array([r["spam_score"] for r in results], dtype=np.float64)

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    if args.eval_mode == "val":
        thresh, prec, rec, f1 = find_threshold_max_f1_min_precision(
            y_true, y_proba, min_precision=MIN_PRECISION
        )
        print(f"\nПересчитанный порог (max F1 при precision >= {MIN_PRECISION}):")
        print(f"  Порог: {thresh:.4f}")
        print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        y_pred = (y_proba >= thresh).astype(np.int64)
    else:
        thresh = (
            float(args.threshold)
            if args.threshold is not None
            else float(getattr(model, "optimal_threshold", 0.5))
        )
        y_pred = (y_proba >= thresh).astype(np.int64)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"\nTest-eval: используем фиксированный порог = {thresh:.4f}")
        print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"  Average Precision (AP): {ap:.4f}")
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    default_errors = "val_errors.csv" if args.eval_mode == "val" else "test_errors.csv"
    errors_path = args.errors_output or str(model_dir / default_errors)
    Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in fp_idx:
        text = texts[idx]
        feats = extract_spam_features(text)
        rows.append({"error_type": "FP", "text": text, "spam_score": float(y_proba[idx]), **feats})
    for idx in fn_idx:
        text = texts[idx]
        feats = extract_spam_features(text)
        rows.append({"error_type": "FN", "text": text, "spam_score": float(y_proba[idx]), **feats})
    if rows:
        err_df = pd.DataFrame(rows)
        cols = ["error_type", "text", "spam_score"] + SPAM_FEATURE_NAMES
        err_df = err_df[[c for c in cols if c in err_df.columns]]
        err_df.to_csv(errors_path, index=False, encoding="utf-8")
        print(f"\nПримеры ошибок (FP: {len(fp_idx)}, FN: {len(fn_idx)}) сохранены: {errors_path}")
    else:
        print("\nОшибок FP/FN нет, файл не создаётся.")

    importances_path = args.importances_output or str(model_dir / "feature_importances.csv")
    Path(importances_path).parent.mkdir(parents=True, exist_ok=True)
    coef = model.model.coef_[0]
    tfidf_names = model.vectorizer.get_feature_names_out()
    all_names = list(tfidf_names) + list(SPAM_FEATURE_NAMES)
    imp_df = pd.DataFrame({
        "feature": all_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    })
    imp_df = imp_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    imp_df.to_csv(importances_path, index=False, encoding="utf-8")
    print(f"Важность признаков сохранена: {importances_path}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_curve, precision_curve, color="navy", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.axhline(MIN_PRECISION, color="gray", linestyle="--", alpha=0.7, label=f"Precision = {MIN_PRECISION}")
    ax.scatter([rec], [prec], color="red", s=80, zorder=5, label=f"Порог {thresh:.3f} (F1={f1:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall на валидации (спам TF-IDF)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    if args.output:
        out_path = args.output
    else:
        out_path = str(model_dir / ("pr_curve.png" if args.eval_mode == "val" else "pr_curve_test.png"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик сохранён: {out_path}")

    if args.update_params:
        if args.eval_mode != "val":
            print("  update-params отключен для eval-mode='test' (порог не пересчитываем).")
            return
        params_path = model_dir / "params.json"
        if not params_path.exists():
            print("  params.json не найден, пропуск обновления.")
        else:
            with open(params_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            old_th = meta.get("optimal_threshold")
            meta["optimal_threshold"] = thresh
            meta["threshold_criterion"] = f"max_f1_min_precision_{MIN_PRECISION}"
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"  params.json обновлён: optimal_threshold {old_th} -> {thresh:.4f}")


def _run_regex(args, df: pd.DataFrame, texts: list, y_true: np.ndarray, batch_size: int) -> None:
    from app.models.spam.regex_model import SpamRegexModel

    print("Загрузка regex спам-модели (pre-filter)...")
    model = SpamRegexModel()
    model.load()
    print("Предсказания: только regex-паттерны (earnings, cta_links, casino, crypto, ...).")

    results = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Получение предсказаний regex...",
    ):
        batch = texts[start : start + batch_size]
        results.extend(model.predict_batch(batch))
    print()

    y_pred = np.array([1 if r["is_spam"] else 0 for r in results], dtype=np.int64)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nRegex (бинарный вывод, порог не используется):")
    print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    out_dir = Path(args.errors_output).parent if args.errors_output else Path("models/spam/regex")
    default_errors = "val_errors_regex.csv" if args.eval_mode == "val" else "test_errors_regex.csv"
    errors_path = args.errors_output or str(out_dir / default_errors)
    Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in fp_idx:
        r = results[idx]
        cats = r.get("spam_categories") or {}
        rows.append({
            "error_type": "FP",
            "text": texts[idx],
            "matched_categories": ",".join(sorted(cats.keys())),
            **{f"cat_{k}": 1 for k in cats},
        })
    for idx in fn_idx:
        rows.append({
            "error_type": "FN",
            "text": texts[idx],
            "matched_categories": "",
        })
    if rows:
        err_df = pd.DataFrame(rows)
        err_df.to_csv(errors_path, index=False, encoding="utf-8")
        print(f"\nОшибки regex (FP: {len(fp_idx)}, FN: {len(fn_idx)}) сохранены: {errors_path}")
    else:
        print("\nОшибок FP/FN нет, файл не создаётся.")

    print("\nИтого (regex):")
    print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"  FP: {len(fp_idx)}, FN: {len(fn_idx)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Валидация спам-модели (tfidf или regex) на val.parquet"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("tfidf", "regex"),
        default="tfidf",
        help="Тип модели: tfidf (PR-кривая, порог, признаки) или regex (оценка регулярок)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/spam/tfidf",
        help="Директория TF-IDF модели (model.pkl, vectorizer.pkl); для regex не используется",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="spam_data/val.parquet",
        help="Parquet/CSV с колонками text, label",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Макс. число примеров (по умолчанию все)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Размер батча для predict_batch (tfidf, regex; по умолчанию 256)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="График PR (только tfidf; по умолчанию <model-dir>/pr_curve.png)",
    )
    parser.add_argument(
        "--update-params",
        action="store_true",
        help="Обновить optimal_threshold в params.json (только tfidf)",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=("val", "test"),
        default="val",
        help="Режим оценки: 'val' — подбираем порог (старое поведение), 'test' — используем сохранённый порог model.optimal_threshold без подбора.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Явно заданный порог для is_spam (актуально для eval-mode='test').",
    )
    parser.add_argument(
        "--errors-output",
        type=str,
        default=None,
        help="Файл FP/FN (tfidf: val_errors.csv, regex: val_errors_regex.csv)",
    )
    parser.add_argument(
        "--importances-output",
        type=str,
        default=None,
        help="Файл важности признаков (только tfidf)",
    )
    args = parser.parse_args()

    val_path = Path(args.val_data)
    if not val_path.exists():
        raise FileNotFoundError(f"Валидационные данные не найдены: {val_path}")

    print(f"Загрузка валидации из {args.val_data}...")
    if val_path.suffix.lower() == ".csv":
        df = pd.read_csv(val_path)
    else:
        df = pd.read_parquet(val_path, columns=["text", "label"])
    if args.max_samples is not None:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
        print(f"Подвыборка: {len(df)} примеров (--max-samples={args.max_samples})")
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].values
    if y_true.dtype == bool or np.issubdtype(y_true.dtype, np.bool_):
        y_true = y_true.astype(np.int64)
    else:
        y_true = (np.asarray(y_true) != 0).astype(np.int64)
    print(f"Примеров: {len(texts)} (spam: {int(y_true.sum())})")

    if args.model_type == "tfidf":
        model_dir = Path(args.model_dir)
        if not (model_dir / "model.pkl").exists() or not (model_dir / "vectorizer.pkl").exists():
            raise FileNotFoundError(
                f"В {model_dir} должны быть model.pkl и vectorizer.pkl. Укажите --model-dir."
            )
        _run_tfidf(args, model_dir, df, texts, y_true, args.batch_size)
    else:
        _run_regex(args, df, texts, y_true, args.batch_size)


if __name__ == "__main__":
    main()
