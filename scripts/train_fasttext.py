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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import optuna
from optuna.samplers import TPESampler

# Добавляем путь к app для импорта
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.preprocessing.text_processor import TextProcessor


class FastTextModelTrainer:
    """Класс для обучения FastText модели с оптимизацией гиперпараметров"""
    
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
    
    def objective(self, trial, X, y):
        """
        Objective функция для Optuna
        
        Args:
            trial: Optuna trial
            X: Тексты
            y: Метки
        
        Returns:
            Средний F1 по кросс-валидации
        """
        # Параметры для FastText
        fasttext_params = {
            'dim': trial.suggest_int('dim', 50, 300, step=50),
            'epoch': trial.suggest_int('epoch', 5, 50),
            'lr': trial.suggest_float('lr', 0.01, 1.0, log=True),
            'word_ngrams': trial.suggest_int('word_ngrams', 1, 3),
        }
        
        # Кросс-валидация
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Обучаем модель на обучающих данных
            model = self._train_fasttext_model(
                X_train.tolist(),
                y_train.tolist(),
                **fasttext_params
            )
            
            # Предсказываем вероятности на валидационных данных
            labels, probas = model.predict(X_val.tolist())
            y_pred = [1 if l[0].endswith('1') else 0 for l in labels]
            # y_proba = [p[0] if l else 1 - p[0] for l, p in zip(y_pred, probas)]
            
            # Вычисляем F1
            score = f1_score(y_val, y_pred)            
            scores.append(score)
            
            # Очищаем память
            del model
        
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
        
        print(f"\nЛучший F1: {self.best_score:.4f}")
        print(f"Лучшие параметры:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        # Обучаем финальную модель на всех данных
        print("\nОбучение финальной модели на всех данных...")
        
        # Восстанавливаем параметры FastText
        fasttext_params = {
            'dim': self.best_params['dim'],
            'epoch': self.best_params['epoch'],
            'lr': self.best_params['lr'],
            'word_ngrams': self.best_params['word_ngrams'],
        }
        
        # Обучаем финальную FastText модель на всех данных
        self.best_model = self._train_fasttext_model(
            X.tolist(),
            y.tolist(),
            **fasttext_params
        )
        
        # Предсказываем вероятности на всех данных
        labels, probas = self.best_model.predict(X.tolist())
        y_pred = [1 if l[0].endswith('1') else 0 for l in labels]
        y_proba = [p[0] if l else 1 - p[0] for l, p in zip(y_pred, probas)]
        
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_proba)
        
        print(f"\nМетрики на всех данных:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        
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
                'best_score_metric': 'roc_auc',
                'n_folds': self.n_folds,
                'random_state': self.random_state
            }
            with open(params_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Параметры сохранены: {params_path}")


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Обучение FastText модели')
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
        default='models/fasttext',
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
    
    # Создание trainer
    trainer = FastTextModelTrainer(
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
        model_path=str(output_dir / 'fasttext_model.bin'),
        params_path=str(output_dir / 'params.json')
    )
    
    print("\nОбучение завершено!")


if __name__ == '__main__':
    main()