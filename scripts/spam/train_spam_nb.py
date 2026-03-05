"""Обучение спам-модели на наивном Байесе (ComplementNB) поверх TF-IDF. Без ручных признаков."""
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Optuna не установлен. Установите: pip install optuna")

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.shared.cli import (
    add_common_data_args,
    add_common_optuna_args,
    add_common_output_arg,
    add_common_random_state_arg,
)
from scripts.shared.common import find_threshold_max_f1_min_precision
from scripts.shared.data import load_train_val_data, prepare_texts_spam

MIN_PRECISION_FOR_THRESHOLD = 0.90


class SpamNBTrainer:
    """TF-IDF + ComplementNB, без ручных признаков. Optuna подбирает параметры TF-IDF и alpha."""

    def __init__(
        self,
        n_folds: int = 5,
        n_trials: int = 50,
        use_cv: bool = True,
        random_state: int = 42,
    ):
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.use_cv = use_cv
        self.random_state = random_state
        self.best_params = None
        self.best_model = None
        self.best_vectorizer = None
        self.best_score = None
        self.optimal_threshold = 0.5

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
    ):
        return prepare_texts_spam(df, text_col=text_col, label_col=label_col)

    def get_objective_score(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
    ) -> float:
        vec = TfidfVectorizer(**params["tfidf_params"])
        X_train_tfidf = vec.fit_transform(X_train)
        X_val_tfidf = vec.transform(X_val)
        model = ComplementNB(alpha=params["alpha"])
        model.fit(X_train_tfidf, y_train)
        y_proba = model.predict_proba(X_val_tfidf)[:, 1]
        _, _, _, score = find_threshold_max_f1_min_precision(
            y_val, y_proba, min_precision=MIN_PRECISION_FOR_THRESHOLD
        )
        del model
        return score

    def objective(
        self,
        trial: "optuna.Trial",
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> float:
        min_ngram = trial.suggest_int("min_ngram", 2, 2)
        max_ngram = trial.suggest_int("max_ngram", 4, 4)
        if min_ngram > max_ngram:
            min_ngram, max_ngram = max_ngram, min_ngram
        tfidf_params = {
            "max_features": trial.suggest_int("max_features", 20000, 60000, step=5000),
            "ngram_range": (min_ngram, max_ngram),
            "min_df": trial.suggest_int("min_df", 5, 30),
            "max_df": trial.suggest_float("max_df", 0.3, 0.7),
            "sublinear_tf": True,
            "analyzer": "char_wb",
            "lowercase": True,
            "dtype": np.float32,
        }
        alpha = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
        params = {
            "tfidf_params": tfidf_params,
            "alpha": alpha,
        }

        if self.use_cv:
            cv = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_t = X_train[train_idx]
                X_v = X_train[val_idx]
                y_t = y_train[train_idx]
                y_v = y_train[val_idx]
                scores.append(self.get_objective_score(X_t, X_v, y_t, y_v, params))
            return float(np.mean(scores))
        if X_val is None or y_val is None:
            raise ValueError("X_val, y_val нужны при use_cv=False")
        return self.get_objective_score(X_train, X_val, y_train, y_val, params)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        study_name: Optional[str] = None,
    ) -> Tuple[Any, Any, Dict]:
        print(f"\nОптимизация: {self.n_trials} trials (TF-IDF + ComplementNB)...")
        if self.use_cv:
            print(f"Кросс-валидация: {self.n_folds} фолдов\n")
        else:
            print("Train/val split\n")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            study_name=study_name or "spam_nb",
        )
        study.optimize(
            lambda t: self.objective(t, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        self.best_params = study.best_params
        self.best_score = study.best_value
        print(f"\nЛучший F1 (при precision >= {MIN_PRECISION_FOR_THRESHOLD}): {self.best_score:.4f}")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")

        print("\nОбучение финальной модели (TF-IDF + ComplementNB)...")
        min_ng, max_ng = self.best_params["min_ngram"], self.best_params["max_ngram"]
        if min_ng > max_ng:
            min_ng, max_ng = max_ng, min_ng
        tfidf_params = {
            "max_features": self.best_params["max_features"],
            "ngram_range": (min_ng, max_ng),
            "min_df": self.best_params["min_df"],
            "max_df": self.best_params["max_df"],
            "sublinear_tf": True,
            "analyzer": "char_wb",
            "lowercase": True,
            "dtype": np.float32,
        }
        self.best_vectorizer = TfidfVectorizer(**tfidf_params)
        X_train_tfidf = self.best_vectorizer.fit_transform(X_train)
        self.best_model = ComplementNB(alpha=self.best_params["alpha"])
        self.best_model.fit(X_train_tfidf, y_train)

        if X_val is not None and y_val is not None:
            X_val_tfidf = self.best_vectorizer.transform(X_val)
            y_proba = self.best_model.predict_proba(X_val_tfidf)[:, 1]
            y_true = y_val
        else:
            y_proba = self.best_model.predict_proba(X_train_tfidf)[:, 1]
            y_true = y_train

        self.optimal_threshold, opt_prec, opt_rec, opt_f1 = find_threshold_max_f1_min_precision(
            y_true, y_proba, min_precision=MIN_PRECISION_FOR_THRESHOLD
        )
        y_pred_opt = (y_proba >= self.optimal_threshold).astype(int)
        roc = roc_auc_score(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        y_pred_05 = (y_proba >= 0.5).astype(int)
        p05 = precision_score(y_true, y_pred_05, zero_division=0)
        r05 = recall_score(y_true, y_pred_05, zero_division=0)
        f05 = f1_score(y_true, y_pred_05, zero_division=0)

        print(f"\nМетрики на валидации:")
        print(f"  ROC-AUC: {roc:.4f}, Average Precision: {ap:.4f}")
        print(f"  Порог 0.5: Precision={p05:.4f}, Recall={r05:.4f}, F1={f05:.4f}")
        print(
            f"  Порог (max F1 при precision>={MIN_PRECISION_FOR_THRESHOLD}) {self.optimal_threshold:.4f}: "
            f"Precision={opt_prec:.4f}, Recall={opt_rec:.4f}, F1={opt_f1:.4f}"
        )

        return self.best_model, self.best_vectorizer, self.best_params

    def save_model(
        self,
        model_path: str,
        vectorizer_path: str,
        params_path: Optional[str] = None,
    ) -> None:
        if self.best_model is None or self.best_vectorizer is None:
            raise ValueError("Сначала вызовите train()")
        out_dir = Path(model_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.best_model, f)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.best_vectorizer, f)
        print(f"Модель: {model_path}, Векторизатор: {vectorizer_path}")
        if params_path:
            min_ng = self.best_params.get("min_ngram")
            max_ng = self.best_params.get("max_ngram")
            ngram_range = [min_ng, max_ng] if min_ng is not None and max_ng is not None else None
            meta = {
                "best_params": self.best_params,
                "best_score": self.best_score,
                "best_score_metric": f"f1_at_min_precision_{MIN_PRECISION_FOR_THRESHOLD}",
                "min_precision_for_threshold": MIN_PRECISION_FOR_THRESHOLD,
                "optimal_threshold": self.optimal_threshold,
                "n_folds": self.n_folds,
                "random_state": self.random_state,
                "classifier": "ComplementNB",
                "use_extra_features": False,
                "tfidf_analyzer": "char_wb",
                "tfidf_ngram_range": ngram_range,
                "tfidf_lowercase": True,
            }
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"Параметры: {params_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение спам-модели на ComplementNB поверх TF-IDF (без ручных признаков)"
    )
    add_common_data_args(parser)
    add_common_optuna_args(parser)
    add_common_output_arg(parser, default_output_dir="models/spam_nb")
    add_common_random_state_arg(parser)
    args = parser.parse_args()

    df_train, df_val, use_cv = load_train_val_data(
        data_path=args.data,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )
    frac = 0.6
    df_train = df_train.sample(frac=frac, random_state=args.random_state)

    trainer = SpamNBTrainer(
        n_folds=args.n_folds,
        n_trials=args.n_trials,
        use_cv=use_cv,
        random_state=args.random_state,
    )
    X_train, y_train = trainer.prepare_data(df_train)
    X_val, y_val = None, None
    if df_val is not None:
        X_val, y_val = trainer.prepare_data(df_val)

    trainer.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        study_name=args.study_name,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(
        model_path=str(out_dir / "model.pkl"),
        vectorizer_path=str(out_dir / "vectorizer.pkl"),
        params_path=str(out_dir / "params.json"),
    )
    print("\nОбучение спам-модели (Naive Bayes) завершено.")


if __name__ == "__main__":
    main()
