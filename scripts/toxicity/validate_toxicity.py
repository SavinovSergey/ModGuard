"""Валидация моделей токсичности на отложенных данных.

Загружает выбранную модель (bert, rnn, fasttext, tfidf, regex),
прогоняет predict_batch, строит Precision-Recall кривую, подбирает
оптимальный порог (max F1 при precision >= MIN_PRECISION), сохраняет:
  — график PR-кривой (pr_curve.png),
  — примеры FP/FN (val_errors.csv).

Для regex-модели (бинарная 0/1) PR-кривая не строится,
выводятся только precision/recall/F1 при фиксированном пороге 0.5.

Запуск:
  python scripts/toxicity/validate_toxicity.py --model-type bert --val-data data/val.parquet
  python scripts/toxicity/validate_toxicity.py --model-type regex --val-data data/val.parquet

Требуется: pip install matplotlib
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.shared.common import find_threshold_max_f1_min_precision

MIN_PRECISION = 0.90

MODEL_TYPES = ("bert", "rnn", "fasttext", "tfidf", "regex")


def _load_model(model_type: str, model_dir: str | None):
    """Загружает модель по типу. Возвращает экземпляр модели."""
    if model_type == "regex":
        from app.models.toxicity.regex_model import RegexModel
        model = RegexModel()
        model.load()
        return model

    if model_type == "tfidf":
        from app.models.toxicity.tfidf_model import TfidfModel
        model_path = Path(model_dir or "models/toxicity/tfidf")
        if not (model_path / "model.pkl").exists():
            raise FileNotFoundError(f"model.pkl не найден в {model_path}")
        model = TfidfModel()
        model.load(
            model_path=str(model_path / "model.pkl"),
            vectorizer_path=str(model_path / "vectorizer.pkl"),
        )
        return model

    if model_type == "fasttext":
        from app.models.toxicity.fasttext_model import FastTextModel
        model_path = Path(model_dir or "models/toxicity/fasttext")
        bin_path = model_path / "fasttext_model.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"fasttext_model.bin не найден в {model_path}")
        model = FastTextModel()
        model.load(model_path=str(bin_path))
        return model

    if model_type == "rnn":
        from app.models.toxicity.rnn_model import RNNModel
        rnn_dir = Path(model_dir or "models/toxicity/rnn")
        tokenizer_path = rnn_dir / "tokenizer.json"
        weights_path = rnn_dir / "model_quantized.pt"
        if not weights_path.exists():
            weights_path = rnn_dir / "model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"model.pt / model_quantized.pt не найден в {rnn_dir}")
        model = RNNModel()
        model.load(model_path=str(weights_path), tokenizer_path=str(tokenizer_path))
        return model

    if model_type == "bert":
        from app.models.toxicity.bert_model import BERTModel
        bert_dir = Path(model_dir or "models/toxicity/bert")
        candidates = [bert_dir, bert_dir / "onnx_cpu", bert_dir / "onnx"]
        for d in candidates:
            if d.is_dir():
                try:
                    model = BERTModel()
                    model.load(model_path=str(d))
                    print(f"BERT загружен из {d}")
                    return model
                except Exception as e:
                    print(f"  Не удалось загрузить из {d}: {e}")
        raise FileNotFoundError(f"BERT модель не найдена (проверялись: {', '.join(str(c) for c in candidates)})")

    raise ValueError(f"Неизвестный тип модели: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Валидация модели токсичности на val-данных: PR-кривая, порог, FP/FN"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=MODEL_TYPES,
        help="Тип модели для валидации",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Директория с артефактами модели (по умолчанию models/<model-type>)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Parquet/CSV с колонками text, label",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Макс. число примеров (для быстрого прогона)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Каталог для артефактов (по умолчанию models/<model-type>)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Размер батча для predict_batch (по умолчанию 256)",
    )
    args = parser.parse_args()

    model_type = args.model_type
    output_dir = Path(args.output_dir or args.model_dir or f"models/toxicity/{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка модели '{model_type}'...")
    model = _load_model(model_type, args.model_dir)
    print(f"Модель '{model_type}' загружена.")

    val_path = Path(args.val_data)
    print(f"Загрузка валидации из {val_path}...")
    if val_path.suffix == ".csv":
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
    print(f"Примеров: {len(texts)} (toxic: {int(y_true.sum())})")

    batch_size = args.batch_size
    results = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Получение предсказаний {model_type}..."):
        batch = texts[start : start + batch_size]
        results.extend(model.predict_batch(batch))
        # done = min(start + batch_size, len(texts))
    print()

    y_proba = np.array([r.get("toxicity_score", 0.0) for r in results], dtype=np.float64)

    is_binary = model_type == "regex"
    if is_binary:
        thresh = 0.5
        y_pred = (y_proba >= thresh).astype(np.int64)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"\nRegex (порог фиксирован = {thresh}):")
        print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    else:
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)

        thresh, prec, rec, f1 = find_threshold_max_f1_min_precision(
            y_true, y_proba, min_precision=MIN_PRECISION
        )
        print(f"\nОптимальный порог (max F1 при precision >= {MIN_PRECISION}):")
        print(f"  Порог: {thresh:.4f}")
        print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"  Average Precision (AP): {ap:.4f}")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall_curve, precision_curve, color="navy", lw=2, label=f"PR (AP = {ap:.3f})")
        ax.axhline(MIN_PRECISION, color="gray", linestyle="--", alpha=0.7, label=f"Precision = {MIN_PRECISION}")
        ax.scatter([rec], [prec], color="red", s=80, zorder=5, label=f"Порог {thresh:.3f} (F1={f1:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall на валидации ({model_type})")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()

        pr_path = str(output_dir / "pr_curve.png")
        fig.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"PR-кривая сохранена: {pr_path}")

    y_pred = (y_proba >= thresh).astype(np.int64)
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    errors_path = str(output_dir / "val_errors.csv")
    rows = []
    for idx in fp_idx:
        rows.append({
            "error_type": "FP",
            "text": texts[idx],
            "toxicity_score": float(y_proba[idx]),
            "true_label": int(y_true[idx]),
        })
    for idx in fn_idx:
        rows.append({
            "error_type": "FN",
            "text": texts[idx],
            "toxicity_score": float(y_proba[idx]),
            "true_label": int(y_true[idx]),
        })
    if rows:
        err_df = pd.DataFrame(rows)
        err_df.to_csv(errors_path, index=False, encoding="utf-8")
        print(f"Ошибки (FP: {len(fp_idx)}, FN: {len(fn_idx)}) сохранены: {errors_path}")
    else:
        print("Ошибок FP/FN нет.")

    print("\nИтого:")
    print(f"  Модель:    {model_type}")
    print(f"  Порог:     {thresh:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  FP: {len(fp_idx)}, FN: {len(fn_idx)}")


if __name__ == "__main__":
    main()
