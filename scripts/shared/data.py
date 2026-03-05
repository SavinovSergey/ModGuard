"""Shared data loading and preprocessing helpers for training scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.preprocessing.text_processor import TextProcessor
from app.preprocessing.spam_processor import SpamTextProcessor


def load_train_val_data(
    data_path: str | None,
    train_data_path: str | None,
    val_data_path: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, bool]:
    """
    Load data from either:
      1) --data (single file, CV mode)
      2) --train-data and --val-data (explicit split)

    Returns:
      df_train, df_val, use_cv
    """
    if train_data_path is not None and val_data_path is not None:
        print(f"Loading train data from {train_data_path}...")
        df_train = pd.read_parquet(train_data_path)
        print(f"Loading validation data from {val_data_path}...")
        df_val = pd.read_parquet(val_data_path)
        return df_train, df_val, False

    if data_path is not None:
        print(f"Loading data from {data_path}...")
        df_train = pd.read_parquet(data_path)
        return df_train, None, True

    raise ValueError("Provide either --data or --train-data with --val-data")


def prepare_texts_classical(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess data for TF-IDF/FastText using full text pipeline."""
    print("Preprocessing text...")
    processor = TextProcessor()
    frame = df.copy()
    frame["processed_text"] = frame[text_col].apply(processor.process)
    frame = frame[frame["processed_text"].str.len() > 0]

    x = frame["processed_text"].values
    y = frame[label_col].values

    print(f"Prepared {len(x)} examples")
    print(f"Class distribution: {np.bincount(y)}")
    return x, y


def prepare_texts_neural(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> tuple[list[str], np.ndarray]:
    """Preprocess data for RNN/BERT using normalization only."""
    print("Preprocessing text...")
    processor = TextProcessor(use_lemmatization=False, remove_stopwords=False)
    frame = df.copy()
    frame["processed_text"] = frame[text_col].apply(processor.normalize)
    frame = frame[frame["processed_text"].str.len() > 0]

    x = frame["processed_text"].tolist()
    y = frame[label_col].values.astype(np.float32)

    print(f"Prepared {len(x)} examples")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    print(f"Unique label values: {np.unique(y)}")
    return x, y


def prepare_texts_spam(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    return_raw: bool = False,
):
    """Preprocess data for spam model: SpamTextProcessor (no lowercasing, keep URLs/emails).
    If return_raw=True, returns (x_processed, y, raw_texts) for feature extraction from raw text."""
    print("Preprocessing text (spam pipeline)...")
    processor = SpamTextProcessor()
    frame = df.copy()
    frame["processed_text"] = frame[text_col].astype(str).apply(processor.process)
    frame = frame[frame["processed_text"].str.len() > 0]
    x = frame["processed_text"].values
    y = frame[label_col].values
    # Ensure int 0/1 for sklearn (label may be bool)
    y = np.asarray(y, dtype=np.int64)
    if y.max() == 1 and y.min() == 0:
        pass
    elif np.issubdtype(y.dtype, np.bool_):
        y = y.astype(np.int64)
    else:
        y = (y != 0).astype(np.int64)
    print(f"Prepared {len(x)} examples")
    print(f"Class distribution: {np.bincount(y)}")
    if return_raw:
        raw_texts = frame[text_col].astype(str).values
        return x, y, raw_texts
    return x, y
