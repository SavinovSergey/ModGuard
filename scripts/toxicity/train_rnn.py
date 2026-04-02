"""Скрипт для обучения RNN модели"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from tqdm import tqdm

# Корень проекта (скрипт в scripts/toxicity/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.models.toxicity.rnn_network import RNNClassifier
from app.models.toxicity.rnn_dataset import ToxicityDataset, collate_fn
from app.models.toxicity.rnn_tokenizers import create_tokenizer, BPETokenizer, RuBERTTokenizer
from scripts.shared.cli import (
    add_common_data_args,
    add_common_loss_args,
    add_common_output_arg,
    add_common_random_state_arg,
)
from scripts.shared.common import (
    BinaryFocalLoss,
    compute_auto_alpha,
    convert_to_json_serializable,
    find_threshold_max_f1_min_precision,
)
from scripts.shared.data import load_train_val_data, prepare_texts_neural
from scripts.shared.bucketing import get_bucket_boundaries, BucketBatchSampler

# Опциональный импорт для квантизации RNN
try:
    from scripts.toxicity.quantize_rnn import quantize_rnn_model
    _HAS_QUANTIZE_RNN = True
except ImportError:
    _HAS_QUANTIZE_RNN = False


class RNNModelTrainer:
    """Класс для обучения RNN модели"""
    
    def __init__(
        self,
        tokenizer_type: str = 'bpe',
        rnn_type: str = 'gru',
        embedding_dim: int = 100,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        max_length: int = 512,
        vocab_size: int = 20000,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 10,
        loss_type: str = 'bce',
        focal_gamma: float = 2.0,
        focal_alpha: float = None,
        focal_auto_alpha: bool = True,
        embedding_dropout: float = 0.0,
        weight_decay: float = 1e-4,
        use_cv: bool = True,
        cv_folds: int = 4,
        use_layer_norm: bool = False,
        max_grad_norm: float = 1.0,
        use_lr_schedule: bool = True,
        lr_schedule_type: str = 'plateau',
        lr_schedule_patience: int = 2,
        lr_schedule_factor: float = 0.5,
        device: str = None,
        random_state: int = 42,
        remove_punctuation: bool = True,
    ):
        """
        Args:
            tokenizer_type: Тип токенизатора ('bpe' или 'rubert')
            rnn_type: Тип RNN ('rnn', 'gru' или 'lstm')
            embedding_dim: Размерность эмбеддингов
            hidden_size: Размерность скрытого состояния
            num_layers: Количество слоев RNN
            dropout: Dropout
            bidirectional: Использовать ли двунаправленную RNN
            max_length: Максимальная длина последовательности
            vocab_size: Размер словаря для BPE
            batch_size: Размер батча
            learning_rate: Learning rate
            epochs: Количество эпох
            loss_type: Тип функции потерь ('bce' или 'focal')
            focal_gamma: Параметр gamma для Focal Loss
            focal_alpha: Параметр alpha для Focal Loss (если None и focal_auto_alpha=True, считается автоматически)
            focal_auto_alpha: Автоматически оценивать alpha по train-сплиту
            embedding_dropout: Dropout для embedding слоя (0.0 = отключен)
            weight_decay: Weight decay для AdamW оптимизатора
            use_cv: Использовать кроссвалидацию вместо train/val split
            cv_folds: Количество фолдов для кроссвалидации
            use_layer_norm: Использовать ли Layer Normalization после RNN
            max_grad_norm: Максимальная норма градиента для clipping (0.0 = отключен)
            use_lr_schedule: Использовать ли learning rate schedule
            lr_schedule_type: Тип scheduler ('plateau' для ReduceLROnPlateau или 'lambda' для LambdaLR)
            lr_schedule_patience: Patience для ReduceLROnPlateau scheduler
            lr_schedule_factor: Фактор уменьшения learning rate для scheduler (для plateau - множитель, для lambda - коэффициент экспоненциального затухания)
            device: Устройство для вычислений
            random_state: Random state для воспроизводимости
            remove_punctuation: Удалять пунктуацию в TextProcessor.normalize (как для классических моделей)
        """
        self.tokenizer_type = tokenizer_type
        self.rnn_type = rnn_type
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_type = loss_type.lower()
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.focal_auto_alpha = focal_auto_alpha
        self.embedding_dropout = embedding_dropout
        self.weight_decay = weight_decay
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.use_layer_norm = use_layer_norm
        self.max_grad_norm = max_grad_norm
        self.use_lr_schedule = use_lr_schedule
        self.lr_schedule_type = lr_schedule_type.lower()
        self.lr_schedule_patience = lr_schedule_patience
        self.lr_schedule_factor = lr_schedule_factor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.remove_punctuation = remove_punctuation

        self.tokenizer = None
        self.model = None
        self.best_model_state = None
        self.best_score = 0.0
        self.loss_alpha_used = None
        self.optimal_threshold = 0.5  # Оптимальный порог будет установлен после обучения
        
        # Устанавливаем seed для воспроизводимости
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Возвращает базовую модель, даже если она обернута в DataParallel."""
        if isinstance(model, nn.DataParallel):
            return model.module
        return model

    @staticmethod
    def _compute_auto_alpha(y_train: np.ndarray) -> float:
        """Backward-compatible wrapper around shared auto-alpha logic."""
        return compute_auto_alpha(y_train)

    @staticmethod
    def _convert_to_json_serializable(obj):
        """Backward-compatible wrapper around shared JSON conversion."""
        return convert_to_json_serializable(obj)

    def prepare_data(self, df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label') -> Tuple[List[str], np.ndarray]:
        """
        Подготовка данных для обучения
        
        Args:
            df: DataFrame с данными
            text_col: Название колонки с текстом
            label_col: Название колонки с метками
        
        Returns:
            X_processed, y
        """
        return prepare_texts_neural(
            df,
            text_col=text_col,
            label_col=label_col,
            remove_punctuation=self.remove_punctuation,
        )
    
    def train_tokenizer(self, texts: List[str]) -> None:
        """
        Обучает BPE токенизатор (если выбран)
        
        Args:
            texts: Список текстов для обучения токенизатора
        """
        if self.tokenizer_type == 'bpe':
            print(f"\nОбучение BPE токенизатора (vocab_size={self.vocab_size})...")
            self.tokenizer = BPETokenizer(vocab_size=self.vocab_size)
            self.tokenizer.train(texts)
            print(f"Токенизатор обучен, размер словаря: {self.tokenizer.get_vocab_size()}")
        elif self.tokenizer_type == 'rubert':
            print(f"\nИспользование ruBERT токенизатора...")
            self.tokenizer = RuBERTTokenizer()
            self.tokenizer._ensure_loaded()
            print(f"Токенизатор загружен, размер словаря: {self.tokenizer.get_vocab_size()}")
        else:
            raise ValueError(f"Неизвестный тип токенизатора: {self.tokenizer_type}")
    
    def train(self, X_train: List[str], y_train: np.ndarray, X_val: List[str] = None, y_val: np.ndarray = None, val_size: float = 0.2) -> None:
        """
        Обучение модели с кроссвалидацией или train/val split
        
        Args:
            X_train: Тексты для обучения
            y_train: Метки для обучения
            X_val: Тексты для валидации (если заданы, используются вместо split)
            y_val: Метки для валидации (если заданы, используются вместо split)
            val_size: Доля валидационной выборки (используется только если use_cv=False и X_val/y_val не заданы)
        """
        print(f"\nНачинаем обучение RNN модели...")
        print(f"Устройство: {self.device}")
        print(f"Тип RNN: {self.rnn_type.upper()}")
        print(f"Тип токенизатора: {self.tokenizer_type.upper()}")
        print(f"Loss: {self.loss_type.upper()}")
        print(f"Параметры: embedding_dim={self.embedding_dim}, hidden_size={self.hidden_size}, "
              f"num_layers={self.num_layers}, dropout={self.dropout}, embedding_dropout={self.embedding_dropout}, "
              f"bidirectional={self.bidirectional}")
        print(f"Оптимизатор: AdamW (weight_decay={self.weight_decay})")
        print(f"Layer Norm: {self.use_layer_norm}, Gradient Clipping: {self.max_grad_norm if self.max_grad_norm > 0 else 'off'}, "
              f"LR Schedule: {self.use_lr_schedule} ({self.lr_schedule_type if self.use_lr_schedule else 'off'})")
        
        if self.use_cv:
            self._train_with_cv(X_train, y_train)
        else:
            if X_val is not None and y_val is not None:
                # Используем переданные валидационные данные
                print(f"\nРазмер обучающей выборки: {len(X_train)}")
                print(f"Размер валидационной выборки: {len(X_val)}")
                self._train_single_fold(X_train, y_train, X_val, y_val, fold_idx=None)
            else:
                # Используем train/val split
                self._train_with_split(X_train, y_train, val_size)
    
    def _train_with_split(self, X: List[str], y: np.ndarray, val_size: float = 0.2) -> None:
        """Обучение с train/val split"""
        # Разделение на train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nРазмер обучающей выборки: {len(X_train)}")
        print(f"Размер валидационной выборки: {len(X_val)}")
        
        self._train_single_fold(X_train, y_train, X_val, y_val, fold_idx=None)
    
    def _train_with_cv(self, X: List[str], y: np.ndarray) -> None:
        """Обучение с кроссвалидацией"""
        print(f"\nИспользуется кроссвалидация на {self.cv_folds} фолдах")
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        fold_scores = []
        fold_best_models = []
        fold_optimal_thresholds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n{'='*60}")
            print(f"Фолд {fold_idx}/{self.cv_folds}")
            print(f"{'='*60}")
            
            X_train = [X[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            print(f"Размер обучающей выборки: {len(X_train)}")
            print(f"Размер валидационной выборки: {len(X_val)}")
            
            # Обучаем модель на этом фолде
            best_score = self._train_single_fold(X_train, y_train, X_val, y_val, fold_idx=fold_idx)
            fold_scores.append(best_score)
            fold_best_models.append(self.best_model_state.copy() if self.best_model_state else None)
            fold_optimal_thresholds.append(self.optimal_threshold)
        
        # Статистика по всем фолдам
        mean_ap = np.mean(fold_scores)
        std_ap = np.std(fold_scores)
        print(f"\n{'='*60}")
        print(f"Результаты кроссвалидации ({self.cv_folds} фолдов):")
        print(f"  Средний Val AP: {mean_ap:.4f} ± {std_ap:.4f}")
        print(f"  Минимальный Val AP: {np.min(fold_scores):.4f}")
        print(f"  Максимальный Val AP: {np.max(fold_scores):.4f}")
        print(f"{'='*60}")
        
        # Выбираем лучшую модель (с максимальным AP)
        best_fold_idx = np.argmax(fold_scores)
        self.best_model_state = fold_best_models[best_fold_idx]
        self.best_score = fold_scores[best_fold_idx]
        self.optimal_threshold = fold_optimal_thresholds[best_fold_idx]
        
        print(f"\nЛучшая модель из фолда {best_fold_idx + 1} (AP: {self.best_score:.4f})")
        print(f"Оптимальный порог для лучшего фолда: {self.optimal_threshold:.4f}")
        
        # Загружаем лучшую модель
        if self.best_model_state:
            self._unwrap_model(self.model).load_state_dict(self.best_model_state)
    
    def _train_single_fold(self, X_train: List[str], y_train: np.ndarray, X_val: List[str], y_val: np.ndarray, fold_idx: int = None) -> float:
        """Обучение модели на одном фолде"""
        # Сохраняем предыдущее состояние best_model_state для кроссвалидации
        prev_best_state = self.best_model_state
        prev_best_score = self.best_score
        
        # Сбрасываем для нового фолда
        self.best_model_state = None
        self.best_score = 0.0
        
        # Сортировка валидации по длине (меньше паддинга на инференсе)
        val_lengths = [len(self.tokenizer.encode(t, max_length=self.max_length)) for t in X_val]
        val_sort_idx = np.argsort(val_lengths)
        X_val = [X_val[j] for j in val_sort_idx]
        y_val = y_val[val_sort_idx]
        
        # Создание датасетов
        train_dataset = ToxicityDataset(X_train, y_train.tolist(), self.tokenizer, max_length=self.max_length)
        val_dataset = ToxicityDataset(X_val, y_val.tolist(), self.tokenizer, max_length=self.max_length)
        
        # Длины для бакетного батчинга (round-robin, неполные батчи отдаём)
        train_lengths = [
            len(self.tokenizer.encode(t, max_length=self.max_length)) for t in X_train
        ]
        bucket_boundaries = get_bucket_boundaries(self.max_length)
        train_sampler = BucketBatchSampler(
            lengths=train_lengths,
            bucket_boundaries=bucket_boundaries,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            seed=self.random_state,
        )
        
        # DataLoader (train — с бакетами, val — без)
        pad_token_id = self.tokenizer.get_pad_token_id()
        collate_with_pad = partial(collate_fn, pad_token_id=pad_token_id)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_with_pad
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_with_pad
        )
        
        # Создание модели
        vocab_size = self.tokenizer.get_vocab_size()
        self.model = RNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type=self.rnn_type,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            padding_idx=pad_token_id,
            embedding_dropout=self.embedding_dropout,
            use_layer_norm=self.use_layer_norm
        ).to(self.device)

        # Включаем DataParallel при наличии нескольких GPU.
        if self.device.startswith('cuda') and torch.cuda.device_count() > 1:
            gpu_count = torch.cuda.device_count()
            print(f"Используется DataParallel на {gpu_count} GPU")
            self.model = nn.DataParallel(self.model)
        
        print(f"\nМодель создана:")
        print(f"  Vocab size: {vocab_size}")
        base_model = self._unwrap_model(self.model)
        print(f"  Параметров: {sum(p.numel() for p in base_model.parameters()):,}")
        
        # Loss и optimizer
        if self.loss_type == 'focal':
            alpha = self.focal_alpha
            if alpha is None and self.focal_auto_alpha:
                alpha = compute_auto_alpha(y_train)
            criterion = BinaryFocalLoss(alpha=alpha, gamma=self.focal_gamma)
            self.loss_alpha_used = alpha
            print(f"  Focal Loss: gamma={self.focal_gamma}, alpha={alpha}")
        elif self.loss_type == 'bce':
            # BCEWithLogitsLoss для бинарной классификации (включает sigmoid + BCE)
            criterion = nn.BCEWithLogitsLoss()
            self.loss_alpha_used = None
        else:
            raise ValueError(f"Неизвестный тип функции потерь: {self.loss_type}. Используйте 'bce' или 'focal'")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        scheduler = None
        if self.use_lr_schedule:
            if self.lr_schedule_type == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',  # Максимизируем Average Precision
                    factor=self.lr_schedule_factor,
                    patience=self.lr_schedule_patience
                )
            elif self.lr_schedule_type == 'lambda':
                # LambdaLR с экспоненциальным затуханием: lr = initial_lr * (factor ^ epoch)
                # Для более плавного затухания используем: lr = initial_lr * (factor ^ (epoch / epochs))
                lr_lambda = lambda epoch: self.lr_schedule_factor ** (epoch / self.epochs)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            else:
                raise ValueError(f"Неизвестный тип scheduler: {self.lr_schedule_type}. Используйте 'plateau' или 'lambda'")
        
        # Обучение
        best_val_ap = 0.0
        optimal_threshold = 0.5  # По умолчанию
        patience = 3
        patience_counter = 0
        
        for epoch in range(self.epochs):
            train_sampler.set_epoch(epoch)
            # Train
            self.model.train()
            train_loss = 0.0
            train_probs = []
            train_labels = []
            
            for batch in tqdm(train_loader, desc="Обучение"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].cpu()

                optimizer.zero_grad()
                logits = self.model(input_ids, lengths=lengths).squeeze(-1)
                loss = criterion(logits, labels)
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                optimizer.step()
                
                train_loss += loss.item()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                train_probs.extend(probs)
                train_labels.extend(labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_auc = roc_auc_score(train_labels, train_probs)
            train_ap = average_precision_score(train_labels, train_probs)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_probs = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Валидация"):
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    lengths = batch['lengths'].cpu()
                    
                    logits = self.model(input_ids, lengths=lengths).squeeze(-1)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_probs.extend(probs)
                    val_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_auc = roc_auc_score(val_labels, val_probs)
            val_ap = average_precision_score(val_labels, val_probs)
            
            # Вычисляем метрики с порогом 0.5 для отображения
            val_preds = (np.array(val_probs) >= 0.5).astype(int)
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_precision = precision_score(val_labels, val_preds, zero_division=0)
            val_recall = recall_score(val_labels, val_preds, zero_division=0)
            
            # Learning rate schedule step
            if scheduler is not None:
                if self.lr_schedule_type == 'plateau':
                    scheduler.step(val_ap)  # Для ReduceLROnPlateau передаем метрику
                elif self.lr_schedule_type == 'lambda':
                    scheduler.step()  # Для LambdaLR просто вызываем step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\nEpoch {epoch + 1}/{self.epochs} (LR: {current_lr:.6f})")
            else:
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train AP: {train_ap:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
            print(f"  Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f} (threshold=0.5)")
            
            # Сохраняем лучшую модель по Average Precision
            if self.best_model_state is None or val_ap > best_val_ap:
                best_val_ap = val_ap
                self.best_score = val_ap
                self.best_model_state = self._unwrap_model(self.model).state_dict()
                patience_counter = 0
                print(f"  ✓ Новая лучшая модель (AP: {val_ap:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping после {epoch + 1} эпох")
                    break
        
        # Загружаем лучшую модель
        if self.best_model_state:
            self._unwrap_model(self.model).load_state_dict(self.best_model_state)
        
        # Подбираем оптимальный порог на валидационных данных
        self.model.eval()
        val_probs_final = []
        val_labels_final = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].cpu()
                logits = self.model(input_ids, lengths=lengths).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs_final.extend(probs)
                val_labels_final.extend(labels.cpu().numpy())
        
        optimal_threshold, opt_precision, opt_recall, opt_f1 = find_threshold_max_f1_min_precision(
            np.array(val_labels_final), np.array(val_probs_final), min_precision=0.9
        )
        
        # Вычисляем финальные метрики с оптимальным порогом
        val_preds_optimal = (np.array(val_probs_final) >= optimal_threshold).astype(int)
        # opt_f1 = f1_score(val_labels_final, val_preds_optimal, zero_division=0)
        
        if fold_idx is None:
            print(f"\nОбучение завершено. Лучший Val AP: {best_val_ap:.4f}")
            print(f"\nОптимальный порог: {optimal_threshold:.4f}")
            print(f"  Precision: {opt_precision:.4f}, Recall: {opt_recall:.4f}, F1: {opt_f1:.4f}")
        
        # Сохраняем оптимальный порог для использования при сохранении модели
        self.optimal_threshold = optimal_threshold
        
        return best_val_ap
    
    def save_model(
        self,
        model_path: str,
        tokenizer_path: str,
        params_path: str = None
    ) -> None:
        """
        Сохранение обученной модели
        
        Args:
            model_path: Путь для сохранения модели
            tokenizer_path: Путь для сохранения токенизатора
            params_path: Путь для сохранения параметров (опционально)
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        if self.tokenizer is None:
            raise ValueError("Токенизатор не обучен. Сначала вызовите train_tokenizer()")
        
        # Создаем директории если нужно
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем модель
        torch.save(self._unwrap_model(self.model).state_dict(), model_path)
        print(f"\nМодель сохранена: {model_path}")
        
        # Сохраняем токенизатор
        self.tokenizer.save(tokenizer_path)
        
        # Сохраняем параметры если указан путь
        if params_path:
            metadata = {
                'tokenizer_type': self.tokenizer_type,
                'rnn_type': self.rnn_type,
                'embedding_dim': self.embedding_dim,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'max_length': self.max_length,
                'vocab_size': self.vocab_size,
                'best_score': self.best_score,
                'best_score_metric': 'average_precision',
                'optimal_threshold': self.optimal_threshold,
                'random_state': self.random_state,
                'remove_punctuation': self.remove_punctuation,
                'loss_type': self.loss_type,
                'focal_gamma': self.focal_gamma,
                'focal_alpha': self.focal_alpha,
                'focal_auto_alpha': self.focal_auto_alpha,
                'embedding_dropout': self.embedding_dropout,
                'weight_decay': self.weight_decay,
                'use_cv': self.use_cv,
                'cv_folds': self.cv_folds if self.use_cv else None,
                'use_layer_norm': self.use_layer_norm,
                'max_grad_norm': self.max_grad_norm,
                'use_lr_schedule': self.use_lr_schedule,
                'lr_schedule_type': self.lr_schedule_type if self.use_lr_schedule else None,
                'lr_schedule_patience': self.lr_schedule_patience if (self.use_lr_schedule and self.lr_schedule_type == 'plateau') else None,
                'lr_schedule_factor': self.lr_schedule_factor if self.use_lr_schedule else None
            }
            # Добавляем model_name для ruBERT токенизатора
            if self.tokenizer_type == 'rubert' and isinstance(self.tokenizer, RuBERTTokenizer):
                metadata['model_name'] = self.tokenizer.model_name
            
            # Преобразуем numpy типы в стандартные Python типы
            metadata = convert_to_json_serializable(metadata)
            
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Параметры сохранены: {params_path}")


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Обучение RNN модели')
    add_common_data_args(parser)
    parser.add_argument(
        '--tokenizer-type',
        type=str,
        choices=['bpe', 'rubert'],
        default='bpe',
        help='Тип токенизатора (bpe или rubert)'
    )
    parser.add_argument(
        '--rnn-type',
        type=str,
        choices=['rnn', 'gru', 'lstm'],
        default='gru',
        help='Тип RNN (rnn, gru или lstm)'
    )
    add_common_output_arg(parser, default_output_dir='models/toxicity/rnn')
    add_common_random_state_arg(parser)
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Количество эпох'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Размер батча'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Максимальная длина последовательности'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=20000,
        help='Размер словаря для BPE токенизатора'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=100,
        help='Размерность эмбеддингов'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=128,
        help='Размерность скрытого состояния'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0003,
        help='Learning rate'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout'
    )
    parser.add_argument(
        '--embedding-dropout',
        type=float,
        default=0.0,
        help='Dropout для embedding слоя (0.0 = отключен)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay для AdamW оптимизатора'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=4,
        help='Количество фолдов для кроссвалидации'
    )
    parser.set_defaults(use_cv=True)
    add_common_loss_args(parser)
    parser.add_argument(
        '--keep-punctuation',
        action='store_true',
        help='Не удалять пунктуацию при подготовке текста (TextProcessor.remove_punkt=False)',
    )
    parser.add_argument(
        '--no-focal-auto-alpha',
        action='store_false',
        dest='focal_auto_alpha',
        help='Не оценивать alpha автоматически'
    )
    parser.set_defaults(focal_auto_alpha=True)
    parser.add_argument(
        '--use-layer-norm',
        action='store_true',
        help='Использовать Layer Normalization после RNN'
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help='Максимальная норма градиента для clipping (0.0 = отключен)'
    )
    parser.add_argument(
        '--use-lr-schedule',
        action='store_true',
        help='Использовать learning rate schedule'
    )
    parser.add_argument(
        '--bidirectional',
        action='store_true',
        help='Двунаправленность RNN'
    )
    parser.set_defaults(bidirectional=False)
    parser.add_argument(
        '--no-lr-schedule',
        action='store_false',
        dest='use_lr_schedule',
        help='Не использовать learning rate schedule'
    )
    parser.set_defaults(use_lr_schedule=True)
    parser.add_argument(
        '--lr-schedule-type',
        type=str,
        choices=['plateau', 'lambda'],
        default='plateau',
        help='Тип learning rate scheduler: plateau (ReduceLROnPlateau) или lambda (LambdaLR с экспоненциальным затуханием)'
    )
    parser.add_argument(
        '--lr-schedule-patience',
        type=int,
        default=2,
        help='Patience для ReduceLROnPlateau scheduler (используется только для --lr-schedule-type plateau)'
    )
    parser.add_argument(
        '--lr-schedule-factor',
        type=float,
        default=0.5,
        help='Фактор уменьшения learning rate: для plateau - множитель при снижении, для lambda - коэффициент экспоненциального затухания'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='После обучения применить динамическую квантизацию модели',
    )
    parser.add_argument(
        '--quantize-output-dir',
        type=str,
        default=None,
        help='Директория для квантизированной модели (по умолчанию: <output_dir>_quantized)',
    )
    parser.add_argument(
        '--quantize-dtype',
        type=str,
        choices=['qint8', 'float16'],
        default='qint8',
        help='Тип квантизации: qint8 (int8) или float16 (FP16)',
    )
    args = parser.parse_args()
    
    df_train, df_val, use_cv = load_train_val_data(
        data_path=args.data,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )
    
    # Создание trainer
    trainer = RNNModelTrainer(
        tokenizer_type=args.tokenizer_type,
        rnn_type=args.rnn_type,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        bidirectional=args.bidirectional,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        dropout=args.dropout,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        focal_auto_alpha=args.focal_auto_alpha,
        embedding_dropout=args.embedding_dropout,
        weight_decay=args.weight_decay,
        use_cv=use_cv,
        cv_folds=args.cv_folds,
        use_layer_norm=args.use_layer_norm,
        max_grad_norm=args.max_grad_norm,
        use_lr_schedule=args.use_lr_schedule,
        lr_schedule_type=args.lr_schedule_type,
        lr_schedule_patience=args.lr_schedule_patience,
        lr_schedule_factor=args.lr_schedule_factor,
        random_state=args.random_state,
        remove_punctuation=not args.keep_punctuation,
    )
    
    # Подготовка данных
    X_train, y_train = trainer.prepare_data(df_train)
    
    if df_val is not None:
        X_val, y_val = trainer.prepare_data(df_val)
    else:
        X_val, y_val = None, None
    
    # Обучение токенизатора (только на обучающих данных)
    trainer.train_tokenizer(X_train)
    
    # Обучение модели
    trainer.train(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # Сохранение
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(
        model_path=str(output_dir / 'model.pt'),
        tokenizer_path=str(output_dir / 'tokenizer.json'),
        params_path=str(output_dir / 'params.json')
    )

    # Опциональная квантизация
    if args.quantize:
        if not _HAS_QUANTIZE_RNN:
            print("\nВнимание: модуль квантизации не найден. Пропуск квантизации.")
            print("Убедитесь, что скрипт quantize_rnn.py доступен.")
        else:
            import torch
            quantize_dir = args.quantize_output_dir or f"{output_dir}_quantized"
            dtype_map = {"qint8": torch.qint8, "float16": torch.float16}
            dtype = dtype_map[args.quantize_dtype]
            print(f"\nПрименение динамической квантизации ({args.quantize_dtype}) -> {quantize_dir}")
            quantize_rnn_model(
                str(output_dir / 'model.pt'),
                str(output_dir / 'tokenizer.json'),
                quantize_dir,
                device=trainer.device,
                dtype=dtype,
            )
    
    print("\nОбучение завершено!")


if __name__ == '__main__':
    main()

