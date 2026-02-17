"""Скрипт для обучения TF-IDF модели с Optuna оптимизацией"""
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

# Проверка наличия optuna
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError(
        "Optuna не установлен. Установите его командой:\n"
        "  pip install optuna\n"
        "или в Jupyter notebook:\n"
        "  !pip install optuna"
    )

# Добавляем путь к app для импорта
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.training.cli import (
    add_common_data_args,
    add_common_optuna_args,
    add_common_output_arg,
    add_common_random_state_arg,
)
from scripts.training.common import find_optimal_threshold
from scripts.training.data import load_train_val_data, prepare_texts_classical


class TfidfModelTrainer:
    """Класс для обучения TF-IDF модели с оптимизацией гиперпараметров"""
    
    def __init__(
        self,
        n_folds: int = 5,
        n_trials: int = 50,
        use_cv: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            n_folds: Количество фолдов для кросс-валидации (используется только если use_cv=True)
            n_trials: Количество trials для Optuna
            use_cv: Использовать кросс-валидацию (True) или train/val split (False)
            random_state: Random state для воспроизводимости
        """
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.use_cv = use_cv
        self.random_state = random_state
        self.best_params = None
        self.best_model = None
        self.best_vectorizer = None
        self.best_score = None
        self.optimal_threshold = 0.5  # Оптимальный порог будет установлен после обучения

    @staticmethod
    def _find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        min_precision: float = 0.9,
    ) -> tuple:
        """Backward-compatible wrapper around shared threshold search."""
        return find_optimal_threshold(y_true, y_proba, min_precision=min_precision)
    
    def prepare_data(self, df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label'):
        """
        Подготовка данных для обучения
        
        Args:
            df: DataFrame с данными
            text_col: Название колонки с текстом
            label_col: Название колонки с метками
        
        Returns:
            X_processed, y
        """
        return prepare_texts_classical(df, text_col=text_col, label_col=label_col)
    
    def get_objective_score(self, X_train, X_val, y_train, y_val, params):
        """
        Функция для обучения модели на одной итерации objective и получения результата

        Args:
            X_train: Тексты для обучения
            y_train: Метки для обучения
            X_val: Тексты для валидации
            y_val: Метки для валидации
            params: Словарь именованных аргументов для модели, векторизатора и т.д.
        Returns:
            Average Precision score по результатам обучения одной модели
        """
        # Создаем и обучаем векторизатор только на обучающих данных
        vectorizer = TfidfVectorizer(**params['tfidf_params'])
        X_train_vectorized = vectorizer.fit_transform(X_train)
                
        # Применяем векторизатор к валидационным данным
        X_val_vectorized = vectorizer.transform(X_val)
                
        # Обучаем модель на обучающих данных
        model = LogisticRegression(**params['lr_params'])
        model.fit(X_train_vectorized, y_train)
                
        # Предсказываем вероятности на валидационных данных для Average Precision
        y_proba = model.predict_proba(X_val_vectorized)[:, 1]
        # Вычисляем Average Precision
        score = average_precision_score(y_val, y_proba)

        del model

        return score
    


    def objective(self, trial, X_train, y_train, X_val=None, y_val=None):
        """
        Objective функция для Optuna
        
        Args:
            trial: Optuna trial
            X_train: Тексты для обучения
            y_train: Метки для обучения
            X_val: Тексты для валидации (используется только если use_cv=False)
            y_val: Метки для валидации (используется только если use_cv=False)
        
        Returns:
            Average Precision score
        
        Важно: Векторизатор обучается только на обучающих данных,
        чтобы избежать утечки данных (data leakage).
        """
        # Параметры для TF-IDF
        tfidf_params = {
            'max_features': trial.suggest_int('max_features', 10000, 100000, step=10000),
            'ngram_range': (
                1,
                trial.suggest_int('max_ngram', 1, 3)
            ),
            'min_df': trial.suggest_int('min_df', 1, 10),
            'max_df': trial.suggest_float('max_df', 0.5, 1.0),
            'sublinear_tf': trial.suggest_categorical('sublinear_tf', [True, False]),
        }
        
        # Параметры для Logistic Regression
        lr_params = {
            'C': trial.suggest_float('C', 0.01, 100.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'liblinear',  # liblinear работает с l1 и l2
            'max_iter': 1000,
            'random_state': self.random_state
        }

        # словарь прочих аргументов
        params = {
            'tfidf_params': tfidf_params,
            'lr_params': lr_params
        }
        
        if self.use_cv:
            # Кросс-валидация с правильным обучением векторизатора
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            scores = []
            
            # Для каждого фолда обучаем векторизатор только на train данных
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_f, X_val_f = X_train[train_idx], X_train[val_idx]
                y_train_f, y_val_f = y_train[train_idx], y_train[val_idx]
                score = self.get_objective_score(X_train_f, X_val_f, y_train_f, y_val_f, params)
                scores.append(score)
            
            return np.mean(scores)
        else:
            # Используем переданные train/val данные
            if X_val is None or y_val is None:
                raise ValueError("X_val и y_val должны быть заданы при use_cv=False")
            
            score = self.get_objective_score(X_train, X_val, y_train, y_val, params)
            return score
    
    def train(self, X_train, y_train, X_val=None, y_val=None, study_name: str = None):
        """
        Обучение модели с оптимизацией гиперпараметров
        
        Args:
            X_train: Тексты для обучения
            y_train: Метки для обучения
            X_val: Тексты для валидации (используется только если use_cv=False)
            y_val: Метки для валидации (используется только если use_cv=False)
            study_name: Имя study для Optuna (опционально)
        
        Returns:
            Лучшие параметры и модель
        """
        print(f"\nНачинаем оптимизацию с {self.n_trials} trials...")
        if self.use_cv:
            print(f"Используем {self.n_folds}-fold кросс-валидацию\n")
        else:
            print(f"Используем train/val split\n")
            if X_val is None or y_val is None:
                raise ValueError("X_val и y_val должны быть заданы при use_cv=False")
        
        # Создаем study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            study_name=study_name
        )
        
        # Оптимизация
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Получаем лучшие параметры
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\nЛучший Average Precision: {self.best_score:.4f}")
        print(f"Лучшие параметры:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        # Обучаем финальную модель на обучающих данных
        print("\nОбучение финальной модели на обучающих данных...")
        
        # Восстанавливаем параметры TF-IDF
        tfidf_params = {
            'max_features': self.best_params['max_features'],
            'ngram_range': (
                1,
                self.best_params['max_ngram']
            ),
            'min_df': self.best_params['min_df'],
            'max_df': self.best_params['max_df'],
            'sublinear_tf': self.best_params['sublinear_tf'],
        }
        
        # Восстанавливаем параметры Logistic Regression
        lr_params = {
            'C': self.best_params['C'],
            'l1_ratio': 0,
            'solver': 'liblinear',
            'max_iter': 1000,
            'random_state': self.random_state,
        }
        
        # Создаем и обучаем финальную модель на обучающих данных
        self.best_vectorizer = TfidfVectorizer(**tfidf_params)
        X_train_vectorized = self.best_vectorizer.fit_transform(X_train)
        
        self.best_model = LogisticRegression(**lr_params)
        self.best_model.fit(X_train_vectorized, y_train)
        
        # Получаем вероятности на валидационных данных для оценки метрик
        if X_val is not None and y_val is not None:
            X_val_vectorized = self.best_vectorizer.transform(X_val)
            y_proba = self.best_model.predict_proba(X_val_vectorized)[:, 1]
            y_true = y_val
        else:
            # Если валидационные данные не заданы, используем обучающие
            y_proba = self.best_model.predict_proba(X_train_vectorized)[:, 1]
            y_true = y_train
        
        # Подбираем оптимальный порог для максимизации recall при Precision >= 90%
        self.optimal_threshold, opt_precision, opt_recall = find_optimal_threshold(
            y_true, y_proba, min_precision=0.9
        )
        
        # Вычисляем метрики с оптимальным порогом
        y_pred_optimal = (y_proba >= self.optimal_threshold).astype(int)
        opt_f1 = f1_score(y_true, y_pred_optimal, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        
        # Также вычисляем метрики с порогом 0.5 для сравнения
        y_pred_default = (y_proba >= 0.5).astype(int)
        default_precision = precision_score(y_true, y_pred_default, zero_division=0)
        default_recall = recall_score(y_true, y_pred_default, zero_division=0)
        default_f1 = f1_score(y_true, y_pred_default, zero_division=0)
        
        print(f"\nМетрики на валидационных данных:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Average Precision: {ap_score:.4f}")
        print(f"\nМетрики с порогом 0.5 (по умолчанию):")
        print(f"  Precision: {default_precision:.4f}, Recall: {default_recall:.4f}, F1: {default_f1:.4f}")
        print(f"\nМетрики с оптимальным порогом ({self.optimal_threshold:.4f}):")
        print(f"  Precision: {opt_precision:.4f}, Recall: {opt_recall:.4f}, F1: {opt_f1:.4f}")
        
        return self.best_model, self.best_vectorizer, self.best_params
    
    def save_model(
        self,
        model_path: str,
        vectorizer_path: str,
        params_path: str = None
    ):
        """
        Сохранение обученной модели
        
        Args:
            model_path: Путь для сохранения модели
            vectorizer_path: Путь для сохранения векторизатора
            params_path: Путь для сохранения параметров (опционально)
        """
        if self.best_model is None or self.best_vectorizer is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        # Создаем директории если нужно
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем модель и векторизатор
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.best_vectorizer, f)
        
        print(f"\nМодель сохранена: {model_path}")
        print(f"Векторизатор сохранен: {vectorizer_path}")
        
        # Сохраняем параметры если указан путь
        if params_path:
            metadata = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'best_score_metric': 'average_precision',
                'optimal_threshold': self.optimal_threshold,
                'n_folds': self.n_folds,
                'random_state': self.random_state
            }
            with open(params_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Параметры сохранены: {params_path}")


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Обучение TF-IDF модели')
    add_common_data_args(parser)
    add_common_optuna_args(parser)
    add_common_output_arg(parser, default_output_dir='models/tfidf')
    add_common_random_state_arg(parser)
    args = parser.parse_args()
    
    df_train, df_val, use_cv = load_train_val_data(
        data_path=args.data,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )
    
    # Создание trainer
    trainer = TfidfModelTrainer(
        n_folds=args.n_folds,
        n_trials=args.n_trials,
        use_cv=use_cv,
        random_state=args.random_state,
    )
    
    # Подготовка данных
    X_train, y_train = trainer.prepare_data(df_train)
    
    if df_val is not None:
        X_val, y_val = trainer.prepare_data(df_val)
    else:
        X_val, y_val = None, None
    
    # Обучение
    trainer.train(X_train, y_train, X_val=X_val, y_val=y_val, study_name=args.study_name)
    
    # Сохранение
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(
        model_path=str(output_dir / 'model.pkl'),
        vectorizer_path=str(output_dir / 'vectorizer.pkl'),
        params_path=str(output_dir / 'params.json')
    )
    
    print("\nОбучение завершено!")


if __name__ == '__main__':
    main()

