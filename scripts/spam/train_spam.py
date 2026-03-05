"""Обучение TF-IDF модели спама с ручными признаками, Optuna и class_weight='balanced'."""
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Optuna не установлен. Установите: pip install optuna")

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.features.spam_features import (
    SPAM_FEATURE_NAMES,
    extract_spam_features_batch,
    matches_caps_word_double_excl_rule,
    matches_caps_word_rule,
)
from scripts.shared.cli import (
    add_common_data_args,
    add_common_optuna_args,
    add_common_output_arg,
    add_common_random_state_arg,
)
from scripts.shared.common import find_threshold_max_f1_min_precision
from scripts.shared.data import load_train_val_data, prepare_texts_spam

# Минимальный precision при подборе порога (max F1 при precision >= MIN_PRECISION)
MIN_PRECISION_FOR_THRESHOLD = 0.90


def _matches_caps_word_any(text: str) -> bool:
    """Срабатывает любое из правил CAPS_WORD (на инференсе такие тексты всегда получат 1)."""
    return matches_caps_word_double_excl_rule(text) or matches_caps_word_rule(text)


def _drop_caps_word_rows(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Исключает строки, подходящие под CAPS_WORD — на них модель не учится."""
    texts = df[text_col].astype(str)
    mask_keep = ~texts.apply(_matches_caps_word_any)
    n_dropped = (~mask_keep).sum()
    if n_dropped > 0:
        print(f"Исключено из выборки по правилам CAPS_WORD: {n_dropped} строк")
    return df.loc[mask_keep].reset_index(drop=True)


def _build_combined(
    X_tfidf,
    raw_texts: np.ndarray,
    scaler: Optional[StandardScaler],
    fit_scaler: bool,
) -> Tuple[Any, Optional[StandardScaler]]:
    """Строит [X_tfidf | X_features]. Если fit_scaler — обучает scaler на фичах."""
    X_feat = extract_spam_features_batch(list(raw_texts))
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(X_feat)
    if scaler is not None:
        X_feat = scaler.transform(X_feat)
    X_feat_sparse = csr_matrix(X_feat.astype(np.float64))
    combined = sparse_hstack([X_tfidf, X_feat_sparse])
    return combined, scaler


class SpamTfidfTrainer:
    """TF-IDF + ручные признаки, Optuna, class_weight='balanced'."""

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
        self.scaler = None
        self.best_score = None
        self.optimal_threshold = 0.5

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        return_raw: bool = False,
    ):
        if return_raw:
            return prepare_texts_spam(df, text_col=text_col, label_col=label_col, return_raw=True)
        return prepare_texts_spam(df, text_col=text_col, label_col=label_col)

    def get_objective_score(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        raw_train: np.ndarray,
        raw_val: np.ndarray,
        params: Dict[str, Any],
    ) -> float:
        vec = TfidfVectorizer(**params["tfidf_params"])
        X_train_tfidf = vec.fit_transform(X_train)
        X_val_tfidf = vec.transform(X_val)
        X_train_combined, scaler = _build_combined(
            X_train_tfidf, raw_train, None, fit_scaler=True
        )
        X_val_combined, _ = _build_combined(X_val_tfidf, raw_val, scaler, fit_scaler=False)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        model.fit(X_train_combined, y_train)
        y_proba = model.predict_proba(X_val_combined)[:, 1]
        _, _, _, f1 = find_threshold_max_f1_min_precision(
            y_val, y_proba, min_precision=MIN_PRECISION_FOR_THRESHOLD
        )
        return float(f1)

    def objective(
        self,
        trial: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        raw_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        raw_val: Optional[np.ndarray] = None,
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
            "lowercase": False,
            "dtype": np.float32,
        }
        params = {"tfidf_params": tfidf_params}

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
                raw_t = raw_train[train_idx]
                raw_v = raw_train[val_idx]
                scores.append(
                    self.get_objective_score(X_t, X_v, y_t, y_v, raw_t, raw_v, params)
                )
            return float(np.mean(scores))
        if X_val is None or y_val is None or raw_val is None:
            raise ValueError("X_val, y_val, raw_val нужны при use_cv=False")
        return self.get_objective_score(
            X_train, X_val, y_train, y_val, raw_train, raw_val, params
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        raw_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        raw_val: Optional[np.ndarray] = None,
        study_name: Optional[str] = None,
    ) -> Tuple[Any, Any, Dict]:
        print(f"\nОптимизация: {self.n_trials} trials (TF-IDF + признаки + LogisticRegression)...")
        if self.use_cv:
            print(f"Кросс-валидация: {self.n_folds} фолдов\n")
        else:
            print("Train/val split\n")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            study_name=study_name or "spam_tfidf",
        )
        study.optimize(
            lambda t: self.objective(t, X_train, y_train, raw_train, X_val, y_val, raw_val),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        self.best_params = study.best_params
        self.best_score = study.best_value
        print(f"\nЛучший F1 (при precision >= {MIN_PRECISION_FOR_THRESHOLD}): {self.best_score:.4f}")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")

        print("\nОбучение финальной модели (TF-IDF + признаки + LogisticRegression)...")
        min_ng = self.best_params["min_ngram"]
        max_ng = self.best_params["max_ngram"]
        if min_ng > max_ng:
            min_ng, max_ng = max_ng, min_ng
        tfidf_params = {
            "max_features": self.best_params["max_features"],
            "ngram_range": (min_ng, max_ng),
            "min_df": self.best_params["min_df"],
            "max_df": self.best_params["max_df"],
            "sublinear_tf": True,
            "analyzer": "char_wb",
            "lowercase": False,
            "dtype": np.float32,
        }
        self.best_vectorizer = TfidfVectorizer(**tfidf_params)
        X_train_tfidf = self.best_vectorizer.fit_transform(X_train)
        X_train_combined, self.scaler = _build_combined(
            X_train_tfidf, raw_train, None, fit_scaler=True
        )
        self.best_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.best_model.fit(X_train_combined, y_train)

        if X_val is not None and y_val is not None and raw_val is not None:
            X_val_tfidf = self.best_vectorizer.transform(X_val)
            X_val_combined, _ = _build_combined(
                X_val_tfidf, raw_val, self.scaler, fit_scaler=False
            )
            y_proba = self.best_model.predict_proba(X_val_combined)[:, 1]
            y_true = y_val
        else:
            y_proba = self.best_model.predict_proba(X_train_combined)[:, 1]
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
        scaler_path: Optional[str] = None,
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
        if self.scaler is not None and scaler_path:
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            print(f"Модель: {model_path}, Векторизатор: {vectorizer_path}, Scaler: {scaler_path}")
        else:
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
                "classifier": "LogisticRegression",
                "use_extra_features": True,
                "spam_feature_names": SPAM_FEATURE_NAMES,
                "tfidf_analyzer": "char_wb",
                "tfidf_ngram_range": ngram_range,
                "tfidf_lowercase": False,
            }
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"Параметры: {params_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение TF-IDF модели спама (TF-IDF + ручные признаки)")
    add_common_data_args(parser)
    add_common_optuna_args(parser)
    add_common_output_arg(parser, default_output_dir="models/spam")
    add_common_random_state_arg(parser)
    args = parser.parse_args()

    df_train, df_val, use_cv = load_train_val_data(
        data_path=args.data,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )
    frac = 0.6
    df_train = df_train.sample(frac=frac, random_state=args.random_state)
    # Исключаем строки, подходящие под CAPS_WORD: на инференсе им всегда выдаём 1, модель учится на остальном
    df_train = _drop_caps_word_rows(df_train)
    if df_val is not None:
        df_val = _drop_caps_word_rows(df_val)
    trainer = SpamTfidfTrainer(
        n_folds=args.n_folds,
        n_trials=args.n_trials,
        use_cv=use_cv,
        random_state=args.random_state,
    )
    X_train, y_train, raw_train = trainer.prepare_data(
        df_train, return_raw=True
    )
    X_val, y_val, raw_val = None, None, None
    if df_val is not None:
        X_val, y_val, raw_val = trainer.prepare_data(df_val, return_raw=True)

    trainer.train(
        X_train,
        y_train,
        raw_train,
        X_val=X_val,
        y_val=y_val,
        raw_val=raw_val,
        study_name=args.study_name,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(
        model_path=str(out_dir / "model.pkl"),
        vectorizer_path=str(out_dir / "vectorizer.pkl"),
        params_path=str(out_dir / "params.json"),
        scaler_path=str(out_dir / "scaler.pkl"),
    )
    print("\nОбучение спам-модели завершено.")


if __name__ == "__main__":
    main()
