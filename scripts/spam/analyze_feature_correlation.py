"""Анализ корреляции ручных признаков (эвристик) для спам-модели."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from app.features.spam_features import SPAM_FEATURE_NAMES, extract_spam_features_batch
from scripts.shared.cli import add_common_data_args
from scripts.shared.data import load_train_val_data, prepare_texts_spam


def _save_correlation_heatmap(corr: pd.DataFrame, path: Path, feature_names: list[str]) -> None:
    """Строит тепловую карту корреляций и сохраняет в файл."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    plt.colorbar(im, ax=ax, label="Корреляция (Pearson)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Строит корреляционную матрицу ручных признаков спама и выводит сильно коррелирующие пары."
    )
    add_common_data_args(parser)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Порог |r| для вывода пар (по умолчанию 0.7)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Путь для сохранения матрицы корреляций в CSV",
    )
    parser.add_argument(
        "--output-heatmap",
        type=str,
        default=None,
        help="Путь для сохранения тепловой карты корреляций (PNG)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Доля данных для анализа (например 0.5). По умолчанию все данные.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state при --sample",
    )
    args = parser.parse_args()

    df_train, df_val, _ = load_train_val_data(
        data_path=args.data,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )
    df = pd.concat([df_train, df_val], ignore_index=True) if df_val is not None else df_train
    if args.sample is not None:
        df = df.sample(frac=args.sample, random_state=args.random_state)
        print(f"Использована выборка: {len(df)} примеров (frac={args.sample})")

    _, _, raw_texts = prepare_texts_spam(df, return_raw=True)
    texts = list(raw_texts)
    print(f"Извлечение признаков для {len(texts)} текстов...")
    X = extract_spam_features_batch(texts)
    feat_df = pd.DataFrame(X, columns=SPAM_FEATURE_NAMES)

    corr = feat_df.corr(method="pearson")
    print("\n--- Матрица корреляций (Pearson) ---")
    print(corr.round(3).to_string())

    print(f"\n--- Пары с |r| >= {args.threshold} ---")
    pairs = []
    for i in range(len(SPAM_FEATURE_NAMES)):
        for j in range(i + 1, len(SPAM_FEATURE_NAMES)):
            r = corr.iloc[i, j]
            if abs(r) >= args.threshold:
                name_i, name_j = SPAM_FEATURE_NAMES[i], SPAM_FEATURE_NAMES[j]
                pairs.append((name_i, name_j, r))
                print(f"  {name_i} vs {name_j}: r = {r:.3f}")
    if not pairs:
        print(f"  Нет пар с |r| >= {args.threshold}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(out_path)
        print(f"\nМатрица корреляций (CSV) сохранена: {out_path}")

    if args.output_heatmap:
        heatmap_path = Path(args.output_heatmap)
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_MATPLOTLIB:
            _save_correlation_heatmap(corr, heatmap_path, SPAM_FEATURE_NAMES)
            print(f"Тепловая карта сохранена: {heatmap_path}")
        else:
            print("matplotlib не установлен, тепловую карту построить нельзя. Установите: pip install matplotlib")


if __name__ == "__main__":
    main()
