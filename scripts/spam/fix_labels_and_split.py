"""Подготовка данных для задачи spam.

Скрипт умеет:
1) загрузить данные из parquet или HuggingFace datasets;
2) адаптировать их к формату text/label;
3) поправить метки по правилам CAPS_WORD;
4) сделать стратифицированный split train/val/test = 60/20/20.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
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


def _normalize_label(value) -> int:
    """Приводит различные форматы меток к 0/1 (1 = spam)."""
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        normalized = value.strip().lower()
        spam_aliases = {"spam", "junk", "toxic", "1", "true", "yes", "y"}
        ham_aliases = {"ham", "not_spam", "0", "false", "no", "n"}
        if normalized in spam_aliases:
            return 1
        if normalized in ham_aliases:
            return 0
    return int(bool(value))


def adapt_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Адаптирует таблицу к единому формату: text(str) + label(0/1)."""
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Ожидаются колонки '{text_col}' и '{label_col}'. "
            f"Найдены: {list(df.columns)}"
        )

    prepared = df[[text_col, label_col]].copy()
    prepared = prepared.rename(columns={text_col: "text", label_col: "label"})
    prepared["text"] = prepared["text"].astype(str).str.strip().str.lower()
    prepared = prepared[prepared["text"] != ""]
    prepared = prepared.drop_duplicates(subset=["text"])
    prepared["label"] = prepared["label"].apply(_normalize_label).astype(int)
    return prepared


def load_source_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """Загружает данные либо из parquet, либо из HuggingFace datasets."""
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")
        print(f"Загрузка данных из parquet: {input_path}...")
        return pd.read_parquet(input_path)

    print(f"Загрузка HuggingFace датасета: {args.hf_dataset}...")
    ds = load_dataset(args.hf_dataset, name=args.hf_config)
    split_name = args.hf_split
    if split_name == "all":
        frames = []
        for split in ds.keys():
            frames.append(ds[split].to_pandas())
        if not frames:
            raise ValueError(f"В датасете {args.hf_dataset} нет доступных split-ов")
        return pd.concat(frames, ignore_index=True)

    if split_name not in ds:
        raise ValueError(f"Split '{split_name}' не найден. Доступно: {list(ds.keys())}")
    return ds[split_name].to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подготовка данных для spam: загрузка, адаптация, фикс меток и split 60/20/20"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        type=str,
        help="Входной parquet файл",
    )
    source_group.add_argument(
        "--hf-dataset",
        type=str,
        help="Имя датасета на HuggingFace, например: community-datasets/sms-spam",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default=None,
        help="Опциональный config/subset HuggingFace датасета",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Какой split брать из HF (train/test/validation/all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/spam",
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

    df = load_source_dataframe(args)
    df = adapt_dataframe(df, text_col=args.text_col, label_col=args.label_col)

    n_before = int((df["label"] != 0).sum())
    df = fix_labels(df, text_col="text", label_col="label")
    n_after = int(df["label"].sum())
    n_fixed = n_after - n_before
    print(f"Меток исправлено по CAPS_WORD: {n_fixed} (спам было {n_before}, стало {n_after})")
    print(f"Всего записей: {len(df)}, спам: {n_after} ({100 * n_after / len(df):.1f}%)")

    # Стратифицированное разбиение 60 / 20 / 20
    df_train, df_rest = train_test_split(
        df,
        test_size=0.4,
        stratify=df["label"],
        random_state=args.random_state,
    )
    df_val, df_test = train_test_split(
        df_rest,
        test_size=0.5,
        stratify=df_rest["label"],
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
