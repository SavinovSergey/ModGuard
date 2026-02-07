"""Скрипт для обучения TF-IDF модели с Optuna оптимизацией"""
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

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
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.preprocessing.text_processor import TextProcessor


class TfidfModelTrainer:
    """Класс для обучения TF-IDF модели с оптимизацией гиперпараметров"""
    
    def __init__(
        self,
        n_folds: int = 5,
        n_trials: int = 50,
        random_state: int = 42
    ):
        """
        Args:
            n_folds: Количество фолдов для кросс-валидации
            n_trials: Количество trials для Optuna
            random_state: Random state для воспроизводимости
        """
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.text_processor = TextProcessor()
        self.best_params = None
        self.best_model = None
        self.best_vectorizer = None
        self.best_score = None
    
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
        # Предобработка текста
        print("Предобработка текста...")
        df = df.copy()
        df['processed_text'] = df[text_col].apply(self.text_processor.process)
        
        # Удаляем пустые тексты после обработки
        df = df[df['processed_text'].str.len() > 0]
        
        X = df['processed_text'].values
        y = df[label_col].values
        
        print(f"Подготовлено {len(X)} примеров")
        print(f"Распределение классов: {np.bincount(y)}")
        
        return X, y
    
    def objective(self, trial, X, y):
        """
        Objective функция для Optuna
        
        Args:
            trial: Optuna trial
            X: Тексты
            y: Метки
        
        Returns:
            Средний ROC-AUC по кросс-валидации
        
        Важно: Векторизатор обучается только на обучающих фолдах,
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
            'l1_ratio': 0,
            # 'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'liblinear',  # liblinear работает с l1 и l2
            'max_iter': 1000,
            'random_state': self.random_state
        }
        
        # Кросс-валидация с правильным обучением векторизатора
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        # Для каждого фолда обучаем векторизатор только на train данных
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Создаем и обучаем векторизатор только на обучающих данных
            vectorizer = TfidfVectorizer(**tfidf_params)
            X_train_vectorized = vectorizer.fit_transform(X_train)
            
            # Применяем векторизатор к валидационным данным
            X_val_vectorized = vectorizer.transform(X_val)
            
            # Обучаем модель на обучающих данных
            model = LogisticRegression(**lr_params)
            model.fit(X_train_vectorized, y_train)
            
            # Предсказываем вероятности на валидационных данных для ROC-AUC
            y_proba = model.predict_proba(X_val_vectorized)[:, 1]
            
            # Вычисляем ROC-AUC
            score = roc_auc_score(y_val, y_proba)
            scores.append(score)
        
        return np.mean(scores)
    
    def train(self, X, y, study_name: str = None):
        """
        Обучение модели с оптимизацией гиперпараметров
        
        Args:
            X: Тексты
            y: Метки
            study_name: Имя study для Optuna (опционально)
        
        Returns:
            Лучшие параметры и модель
        """
        print(f"\nНачинаем оптимизацию с {self.n_trials} trials...")
        print(f"Используем {self.n_folds}-fold кросс-валидацию\n")
        
        # Создаем study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            study_name=study_name
        )
        
        # Оптимизация
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Получаем лучшие параметры
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\nЛучший ROC-AUC: {self.best_score:.4f}")
        print(f"Лучшие параметры:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        # Обучаем финальную модель на всех данных
        print("\nОбучение финальной модели на всех данных...")
        
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
            'class_weight': 'balanced'
        }
        
        # Создаем и обучаем финальную модель
        self.best_vectorizer = TfidfVectorizer(**tfidf_params)
        X_vectorized = self.best_vectorizer.fit_transform(X)
        
        self.best_model = LogisticRegression(**lr_params)
        self.best_model.fit(X_vectorized, y)
        
        # Оценка на всех данных
        y_pred = self.best_model.predict(X_vectorized)
        y_proba = self.best_model.predict_proba(X_vectorized)[:, 1]
        
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_proba)
        
        print(f"\nМетрики на всех данных:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        
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
                'best_score_metric': 'roc_auc',  # Указываем метрику оптимизации
                'n_folds': self.n_folds,
                'random_state': self.random_state
            }
            with open(params_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Параметры сохранены: {params_path}")


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Обучение TF-IDF модели')
    parser.add_argument(
        '--data',
        type=str,
        default='data/train.parquet',
        help='Путь к обучающим данным'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Количество фолдов для кросс-валидации'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Количество trials для Optuna'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/tfidf',
        help='Директория для сохранения модели'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default=None,
        help='Имя study для Optuna'
    )
    args = parser.parse_args()
    
    # Загрузка данных
    print(f"Загрузка данных из {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Подготовка меток (если label == 0, то токсичный)
    if 'label' in df.columns:
        df['label'] = (df['label'] == 0).astype(int)
    
    # Создание trainer
    trainer = TfidfModelTrainer(
        n_folds=args.n_folds,
        n_trials=args.n_trials
    )
    
    # Подготовка данных
    X, y = trainer.prepare_data(df)
    
    # Обучение
    trainer.train(X, y, study_name=args.study_name)
    
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

