"""
Пример использования скрипта обучения FastText модели в ноутбуке

Этот файл можно импортировать в Jupyter notebook для обучения модели
"""

import sys
from pathlib import Path

import optuna

# Добавляем путь к scripts
sys.path.insert(0, str(Path(__file__).parent))

from train_fasttext import FastTextModelTrainer
import pandas as pd


def train_fasttext_model(
    df: pd.DataFrame,
    text_col: str = 'text',
    label_col: str = 'label',
    n_folds: int = 5,
    n_trials: int = 50,
    output_dir: str = 'models/fasttext',
    study_name: str = None,
    is_prepared: bool = False
):
    """
    Обучение FastText модели с оптимизацией гиперпараметров
    
    Args:
        df: DataFrame с данными (должен содержать text_col и label_col)
        text_col: Название колонки с текстом
        label_col: Название колонки с метками (0 или 1, где 1 = токсичный)
        n_folds: Количество фолдов для кросс-валидации
        n_trials: Количество trials для Optuna
        output_dir: Директория для сохранения модели
        study_name: Имя study для Optuna (опционально)
    
    Returns:
        trainer: Обученный FastTextModelTrainer
    """
    
    # Создание trainer
    trainer = FastTextModelTrainer(
        n_folds=n_folds,
        n_trials=n_trials
    )
    
    # Подготовка данных
    if not is_prepared:
        X, y = trainer.prepare_data(df, text_col=text_col, label_col=label_col)
    else:
        X = df[text_col].values
        y = df[label_col].values
    
    # Обучение
    trainer.train(X, y, study_name=study_name)
    
    # Сохранение
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(
        model_path=str(output_path / 'fasttext_model.bin'),
        params_path=str(output_path / 'params.json')
    )
    
    return trainer

