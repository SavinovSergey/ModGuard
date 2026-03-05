"""Скрипт для обучения FastText модели с Optuna оптимизацией"""
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import tempfile

import numpy as np
import pandas as pd
import fasttext
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

import optuna
from optuna.samplers import TPESampler

# Корень проекта (скрипт в scripts/toxicity/)
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.shared.cli import (
    add_common_data_args,
    add_common_optuna_args,
    add_common_output_arg,
    add_common_random_state_arg,
)
from scripts.shared.common import find_optimal_threshold
from scripts.shared.data import load_train_val_data, prepare_texts_classical


class FastTextModelTrainer:
    """Класс для обучения FastText модели с оптимизацией гиперпараметров"""
    
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
    
    def _train_fasttext_model(self, texts: list, labels: list, **kwargs):
        """
        Обучение FastText модели
        
        Args:
            texts: Список текстов
            labels: Список меток (0 или 1)
            **kwargs: Параметры для fasttext.train_supervised
        
        Returns:
            Обученная FastText модель
        """
        # Создаем временный файл для FastText
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            # FastText требует формат: __label__0 текст или __label__1 текст
            for text, label in zip(texts, labels):
                label_str = '__label__1' if label == 1 else '__label__0'
                f.write(f"{label_str} {text}\n")
            temp_file = f.name
        
        # Обучаем модель
        model = fasttext.train_supervised(
            temp_file,
            **kwargs,
            verbose=0  # Отключаем вывод
        )
        # Удаляем временный файл
        os.unlink(temp_file)
        
        return model
    
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
        """
        # Параметры для FastText
        fasttext_params = {
            'dim': trial.suggest_int('dim', 50, 300, step=50),
            'epoch': trial.suggest_int('epoch', 5, 50),
            'lr': trial.suggest_float('lr', 0.01, 1.0, log=True),
            'word_ngrams': trial.suggest_int('word_ngrams', 1, 3),
        }
        
        if self.use_cv:
            # Кросс-валидация
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                
                # Обучаем модель на обучающих данных
                model = self._train_fasttext_model(
                    X_train_fold.tolist(),
                    y_train_fold.tolist(),
                    **fasttext_params
                )
                
                # Предсказываем вероятности на валидационных данных
                labels, probas = model.predict(X_val_fold.tolist())
                # Преобразуем вероятности: если предсказан класс 1, берем его вероятность, иначе 1 - вероятность класса 0
                y_proba = [p[0] if l[0].endswith('1') else 1 - p[0] for l, p in zip(labels, probas)]
                
                # Вычисляем Average Precision
                score = average_precision_score(y_val_fold, y_proba)
                scores.append(score)
                
                # Очищаем память
                del model
            
            return np.mean(scores)
        else:
            # Используем переданные train/val данные
            if X_val is None or y_val is None:
                raise ValueError("X_val и y_val должны быть заданы при use_cv=False")
            
            # Обучаем модель на обучающих данных
            model = self._train_fasttext_model(
                X_train.tolist(),
                y_train.tolist(),
                **fasttext_params
            )
            
            # Предсказываем вероятности на валидационных данных
            labels, probas = model.predict(X_val.tolist())
            # Преобразуем вероятности: если предсказан класс 1, берем его вероятность, иначе 1 - вероятность класса 0
            y_proba = np.array([p[0] if l[0].endswith('1') else 1 - p[0] for l, p in zip(labels, probas)])
            
            # Вычисляем Average Precision
            score = average_precision_score(y_val, y_proba)
            
            # Очищаем память
            del model
            
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
        
        # Восстанавливаем параметры FastText
        fasttext_params = {
            'dim': self.best_params['dim'],
            'epoch': self.best_params['epoch'],
            'lr': self.best_params['lr'],
            'word_ngrams': self.best_params['word_ngrams'],
        }
        
        # Обучаем финальную FastText модель на обучающих данных
        self.best_model = self._train_fasttext_model(
            X_train.tolist(),
            y_train.tolist(),
            **fasttext_params
        )
        
        # Получаем вероятности на валидационных данных для оценки метрик
        if X_val is not None and y_val is not None:
            labels, probas = self.best_model.predict(X_val.tolist())
            y_proba = np.array([p[0] if l[0].endswith('1') else 1 - p[0] for l, p in zip(labels, probas)])
            y_true = y_val
        else:
            # Если валидационные данные не заданы, используем обучающие
            labels, probas = self.best_model.predict(X_train.tolist())
            y_proba = np.array([p[0] if l[0].endswith('1') else 1 - p[0] for l, p in zip(labels, probas)])
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
        
        return self.best_model, self.best_params
    
    def save_model(
        self,
        model_path: str,
        params_path: str = None
    ):
        """
        Сохранение обученной модели
        
        Args:
            model_path: Путь для сохранения классификатора
            params_path: Путь для сохранения параметров (опционально)
        """
        if self.best_model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        # Создаем директории если нужно
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем классификатор
        self.best_model.save_model(model_path)
        print(f"\nКлассификатор сохранен: {model_path}")
        
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
    parser = argparse.ArgumentParser(description='Обучение FastText модели')
    add_common_data_args(parser)
    add_common_optuna_args(parser)
    add_common_output_arg(parser, default_output_dir='models/fasttext')
    add_common_random_state_arg(parser)
    args = parser.parse_args()
    
    df_train, df_val, use_cv = load_train_val_data(
        data_path=args.data,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )
    
    # Создание trainer
    trainer = FastTextModelTrainer(
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
        model_path=str(output_dir / 'fasttext_model.bin'),
        params_path=str(output_dir / 'params.json')
    )
    
    print("\nОбучение завершено!")


if __name__ == '__main__':
    main()