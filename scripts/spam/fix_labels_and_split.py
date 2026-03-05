"""Фикс меток спама по правилам CAPS_WORD и разбиение на train/val/test 60/20/20.

Для записей, в которых текст удовлетворяет хотя бы одному из правил CAPS_WORD
(слово капсом + !! или слово 7+ букв капсом), выставляется метка 1 (спам).
Затем данные стратифицированно разбиваются на train 60%, val 20%, test 20%.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.features.spam_features import (
    matches_caps_word_double_excl_rule,
    matches_caps_word_rule,
)


def _matches_caps_word_any(text: str) -> bool:
    """Срабатывает любое из правил CAPS_WORD."""
    return matches_caps_word_double_excl_rule(text) or matches_caps_word_rule(text)


def fix_labels(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Для записей с текстом, подходящим под CAPS_WORD, ставит метку 1 (спам).
    Остальные метки не меняются.
    """
    df = df.copy()
    texts = df[text_col].astype(str)
    mask = texts.apply(_matches_caps_word_any)
    df.loc[mask, label_col] = 1
    # Нормализуем к 0/1 если были bool или другие значения
    if df[label_col].dtype == bool:
        df[label_col] = df[label_col].astype(int)
    else:
        df[label_col] = (df[label_col] != 0).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Фикс меток по CAPS_WORD и разбиение train/val/test 60/20/20"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Входной parquet с колонками text, label",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="spam_data",
        help="Директория для train.parquet, val.parquet, test.parquet",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Имя колонки с текстом",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Имя колонки с меткой",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state для воспроизводимости разбиения",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    print(f"Загрузка данных из {input_path}...")
    df = pd.read_parquet(input_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"Ожидаются колонки '{args.text_col}' и '{args.label_col}'. "
            f"Найдены: {list(df.columns)}"
        )

    n_before = int((df[args.label_col] != 0).sum())
    df = fix_labels(df, text_col=args.text_col, label_col=args.label_col)
    n_after = int(df[args.label_col].sum())
    n_fixed = n_after - n_before
    print(f"Меток исправлено по CAPS_WORD: {n_fixed} (спам было {n_before}, стало {n_after})")
    print(f"Всего записей: {len(df)}, спам: {n_after} ({100 * n_after / len(df):.1f}%)")

    # Стратифицированное разбиение 60 / 20 / 20
    df_train, df_rest = train_test_split(
        df,
        test_size=0.4,
        stratify=df[args.label_col],
        random_state=args.random_state,
    )
    df_val, df_test = train_test_split(
        df_rest,
        test_size=0.5,
        stratify=df_rest[args.label_col],
        random_state=args.random_state,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    print(f"\nСохранено:")
    print(f"  train: {train_path} ({len(df_train)} записей)")
    print(f"  val:   {val_path} ({len(df_val)} записей)")
    print(f"  test:  {test_path} ({len(df_test)} записей)")


if __name__ == "__main__":
    main()
