"""Валидация спам-модели на отложенных данных (TF-IDF + ручные правила и признаки).

Запускает predict_batch (с правилом caps+!! и ручными признаками),
строит Precision-Recall, подбирает порог (max F1 при precision >= 0.90), сохраняет:
  — график PR-кривой (pr_curve.png),
  — примеры FP/FN с признаками (val_errors.csv),
  — важность признаков (feature_importances.csv).
Опционально обновляет optimal_threshold в params.json.

Требуется: pip install matplotlib
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.features.spam_features import SPAM_FEATURE_NAMES, extract_spam_features
from app.models.spam_tfidf_model import SpamTfidfModel
from scripts.shared.common import find_threshold_max_f1_min_precision

MIN_PRECISION = 0.90


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Валидация спам-модели на val.parquet: PR-кривая, порог (max F1 при precision >= 0.90), FP/FN, важность признаков"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/spam",
        help="Директория с model.pkl, vectorizer.pkl, params.json, scaler.pkl",
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
        help="Макс. число примеров для оценки (для быстрого прогона; по умолчанию все)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения графика (по умолчанию <model-dir>/pr_curve.png)",
    )
    parser.add_argument(
        "--update-params",
        action="store_true",
        help="Обновить optimal_threshold в params.json по новому критерию",
    )
    parser.add_argument(
        "--errors-output",
        type=str,
        default=None,
        help="Файл для примеров FP/FN с признаками (по умолчанию <model-dir>/val_errors.csv)",
    )
    parser.add_argument(
        "--importances-output",
        type=str,
        default=None,
        help="Файл для важности признаков (по умолчанию <model-dir>/feature_importances.csv)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "model.pkl").exists() or not (model_dir / "vectorizer.pkl").exists():
        raise FileNotFoundError(
            f"В {model_dir} должны быть model.pkl и vectorizer.pkl"
        )

    print(f"Загрузка модели из {model_dir}...")
    model = SpamTfidfModel()
    model.load(
        model_path=str(model_dir / "model.pkl"),
        vectorizer_path=str(model_dir / "vectorizer.pkl"),
    )
    print("Предсказания: полный пайплайн (TF-IDF + ручные признаки + правило caps+!!).")

    print(f"Загрузка валидации из {args.val_data}...")
    df = pd.read_parquet(args.val_data, columns=["text", "label"])
    if args.max_samples is not None:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
        print(f"Используется подвыборка: {len(df)} примеров (--max-samples={args.max_samples})")
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].values
    if y_true.dtype == bool or np.issubdtype(y_true.dtype, np.bool_):
        y_true = y_true.astype(np.int64)
    else:
        y_true = (np.asarray(y_true) != 0).astype(np.int64)
    print(f"Примеров: {len(texts)} (spam: {int(y_true.sum())})")

    print("Получение вероятностей...")
    results = model.predict_batch(texts)
    y_proba = np.array([r["spam_score"] for r in results], dtype=np.float64)

    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    thresh, prec, rec, f1 = find_threshold_max_f1_min_precision(
        y_true, y_proba, min_precision=MIN_PRECISION
    )
    print(f"\nПересчитанный порог (max F1 при precision >= {MIN_PRECISION}):")
    print(f"  Порог: {thresh:.4f}")
    print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    y_pred = (y_proba >= thresh).astype(np.int64)
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    # Примеры ошибок FP/FN с ручными признаками
    errors_path = args.errors_output or str(model_dir / "val_errors.csv")
    Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in fp_idx:
        text = texts[idx]
        feats = extract_spam_features(text)
        rows.append({
            "error_type": "FP",
            "text": text,
            "spam_score": float(y_proba[idx]),
            **feats,
        })
    for idx in fn_idx:
        text = texts[idx]
        feats = extract_spam_features(text)
        rows.append({
            "error_type": "FN",
            "text": text,
            "spam_score": float(y_proba[idx]),
            **feats,
        })
    if rows:
        err_df = pd.DataFrame(rows)
        cols = ["error_type", "text", "spam_score"] + SPAM_FEATURE_NAMES
        err_df = err_df[[c for c in cols if c in err_df.columns]]
        err_df.to_csv(errors_path, index=False, encoding="utf-8")
        print(f"\nПримеры ошибок (FP: {len(fp_idx)}, FN: {len(fn_idx)}) сохранены: {errors_path}")
    else:
        print("\nОшибок FP/FN нет, файл не создаётся.")

    # Важность признаков (коэффициенты логистической регрессии)
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

    # Визуализация
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_curve, precision_curve, color="navy", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.axhline(MIN_PRECISION, color="gray", linestyle="--", alpha=0.7, label=f"Precision = {MIN_PRECISION}")
    ax.scatter([rec], [prec], color="red", s=80, zorder=5, label=f"Порог {thresh:.3f} (F1={f1:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall на валидации (спам-модель)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()

    out_path = args.output or str(model_dir / "pr_curve.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик сохранён: {out_path}")

    if args.update_params:
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


if __name__ == "__main__":
    main()
