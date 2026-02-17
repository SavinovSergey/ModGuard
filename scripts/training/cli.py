"""Shared argparse helpers for training scripts."""
from __future__ import annotations

import argparse


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a single parquet dataset (CV mode)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to train parquet dataset",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation parquet dataset",
    )


def add_common_output_arg(parser: argparse.ArgumentParser, default_output_dir: str) -> None:
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Output directory for model artifacts",
    )


def add_common_random_state_arg(parser: argparse.ArgumentParser, default: int = 42) -> None:
    parser.add_argument(
        "--random-state",
        type=int,
        default=default,
        help="Random state for reproducibility",
    )


def add_common_optuna_args(parser: argparse.ArgumentParser, n_folds_default: int = 5, n_trials_default: int = 50) -> None:
    parser.add_argument(
        "--n-folds",
        type=int,
        default=n_folds_default,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=n_trials_default,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional Optuna study name",
    )


def add_common_loss_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["bce", "focal"],
        default="bce",
        help="Loss type",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma for focal loss",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Optional alpha for focal loss",
    )
    parser.add_argument(
        "--focal-auto-alpha",
        action="store_true",
        help="Estimate alpha automatically on train split",
    )
