"""Скрипт для дообучения BERT модели для классификации токсичности"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from tqdm import tqdm

# Корень проекта (скрипт в scripts/toxicity/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

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
    find_optimal_threshold,
    find_threshold_max_f1_min_precision
)
from scripts.shared.data import load_train_val_data, prepare_texts_neural
from scripts.shared.bucketing import get_bucket_boundaries, BucketBatchSampler

# Опциональный импорт для ONNX квантизации
try:
    from scripts.toxicity.quantize_bert_onnx import quantize_bert_to_onnx
    _HAS_QUANTIZE_ONNX = True
except ImportError:
    _HAS_QUANTIZE_ONNX = False


class ToxicityDataset(Dataset):
    """Dataset для токсичности комментариев"""
    
    def __init__(self, texts: List[str], labels: List[float], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


class BERTModelTrainer:
    """Класс для дообучения BERT модели"""
    
    def __init__(
        self,
        model_name: str = 'cointegrated/rubert-tiny2',
        max_length: int = 512,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_steps: Optional[int] = 0,
        warmup_ratio: Optional[float] = 0,
        lr_scheduler_type: str = 'linear',
        no_freeze: bool = True,
        freeze_encoder: bool = True,
        freeze_last_n_layers: int = 0,
        dropout: float = 0.1,
        device: Optional[str] = None,
        loss_type: str = 'bce',
        focal_gamma: Optional[float] = 2,
        focal_alpha: Optional[float] = None,
        focal_auto_alpha: Optional[bool] = True,
        reduction: Optional[str] = "mean",
        max_grad_norm: float = 1.0,
        random_state: int = 42
    ):
        """
        Args:
            model_name: Название модели из HuggingFace
            max_length: Максимальная длина последовательности
            learning_rate: Learning rate
            batch_size: Размер батча
            epochs: Количество эпох
            weight_decay: Weight decay
            warmup_steps: Количество шагов warmup (приоритет над warmup_ratio)
            warmup_ratio: Доля шагов для warmup
            lr_scheduler_type: Тип scheduler (linear, cosine, constant)
            no_freeze: Не замораживать модель вовсе
            freeze_encoder: Замораживать ли энкодер (кроме последнего слоя)
            freeze_last_n_layers: Количество последних слоев для заморозки (0 = разморозить только последний)
            dropout: Dropout для классификационной головы
            device: Устройство для вычислений
            focal_gamma: Приоритет положительному классу
            focal_alpha: Параметр alpha для Focal Loss (если None и focal_auto_alpha=True, считается автоматически)
            focal_auto_alpha: Автоматически оценивать alpha по train-сплиту
            reduction: Агрегация лосса
            max_grad_norm: Максимальная норма градиента для clipping (0.0 = отключен)
            random_state: Random state для воспроизводимости
        """
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.no_freeze = no_freeze
        self.freeze_encoder = freeze_encoder
        self.freeze_last_n_layers = freeze_last_n_layers
        self.dropout = dropout
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_grad_norm = max_grad_norm
        self.random_state = random_state

        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.focal_auto_alpha = focal_auto_alpha
        self.reduction = reduction
        
        self.tokenizer = None
        self.model = None
        self.best_score = 0.0
        self.optimal_threshold = 0.5
        self._best_model_state = None
        
        # Устанавливаем seed для воспроизводимости
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    @staticmethod
    def _find_optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        min_precision: float = 0.9,
    ) -> Tuple[float, float, float]:
        """Backward-compatible wrapper around shared threshold search."""
        return find_optimal_threshold(y_true, y_proba, min_precision=min_precision)

    @staticmethod
    def _compute_auto_alpha(y_train: np.ndarray) -> float:
        """Backward-compatible wrapper around shared auto-alpha logic."""
        return compute_auto_alpha(y_train)

    @staticmethod
    def _convert_to_json_serializable(obj):
        """Backward-compatible wrapper around shared JSON conversion."""
        return convert_to_json_serializable(obj)
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        text_col: str = 'text', 
        label_col: str = 'label'
    ) -> Tuple[List[str], np.ndarray]:
        """
        Подготовка данных для обучения
        
        Args:
            df: DataFrame с данными
            text_col: Название колонки с текстом
            label_col: Название колонки с метками
        
        Returns:
            X_processed, y
        """
        return prepare_texts_neural(df, text_col=text_col, label_col=label_col)
    
    def load_model_and_tokenizer(self):
        """Загружает модель и токенизатор"""
        print(f"\nЗагрузка модели {self.model_name}...")
        
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Загружаем модель для sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,  # Бинарная классификация
            problem_type="single_label_classification"
        )

        if not self.no_freeze:
            #Заморозка эмбеддингов
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
                
            # Замораживаем слои энкодера (кроме последнего)
            if self.freeze_encoder:
                print(f"Заморозка слоев энкодера (кроме последних {self.freeze_last_n_layers + 1} слоев)...")
                
                # Получаем количество слоев
                num_layers = len(self.model.bert.encoder.layer)
                layers_to_freeze = num_layers - 1 - self.freeze_last_n_layers
                
                # Замораживаем все слои кроме последних (freeze_last_n_layers + 1)
                for i in range(layers_to_freeze):
                    for param in self.model.bert.encoder.layer[i].parameters():
                        param.requires_grad = False
                
                print(f"  Заморожено слоев: {layers_to_freeze}")
                print(f"  Обучаемых слоев: {num_layers - layers_to_freeze}")
        
        # Заменяем классификатор на наш с dropout
        hidden_size = self.model.config.hidden_size
        self.model.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Переносим модель на устройство
        self.model = self.model.to(self.device)

        # Включаем DataParallel при наличии нескольких GPU
        if str(self.device).startswith('cuda') and torch.cuda.device_count() > 1:
            gpu_count = torch.cuda.device_count()
            print(f"Используется DataParallel на {gpu_count} GPU")
            self.model = nn.DataParallel(self.model)
        
        print(f"Модель загружена. Устройство: {self.device}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Параметров: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Обучаемых параметров: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str] = None,
        y_val: np.ndarray = None,
        val_size: float = 0.2,
        output_dir: str = 'models/toxicity/bert'
    ) -> None:
        """
        Обучение модели
        
        Args:
            X_train: Тексты для обучения
            y_train: Метки для обучения
            X_val: Тексты для валидации (если заданы, используются вместо split)
            y_val: Метки для валидации (если заданы, используются вместо split)
            val_size: Доля валидационной выборки (используется только если X_val/y_val не заданы)
            output_dir: Директория для сохранения модели
        """
        print(f"\nНачинаем дообучение BERT модели...")
        print(f"Устройство: {self.device}")
        print(f"Модель: {self.model_name}")
        print(f"Параметры: max_length={self.max_length}, batch_size={self.batch_size}, "
              f"learning_rate={self.learning_rate}, epochs={self.epochs}")
        
        # Загружаем модель и токенизатор
        self.load_model_and_tokenizer()
        
        # Разделение на train/val если нужно
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=self.random_state, stratify=y_train
            )
        
        print(f"\nРазмер обучающей выборки: {len(X_train)}")
        print(f"Размер валидационной выборки: {len(X_val)}")
        
        length_batch_size = 512
        # Сортировка валидации по длине (меньше паддинга на инференсе)
        val_lengths = []
        for i in range(0, len(X_val), length_batch_size):
            batch_texts = X_val[i : i + length_batch_size]
            out = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
            )
            val_lengths.extend(out['length'])
        val_sort_idx = np.argsort(val_lengths)
        X_val = [X_val[j] for j in val_sort_idx]
        y_val = y_val[val_sort_idx]
        
        # Создаем датасеты
        train_dataset = ToxicityDataset(X_train, y_train.tolist(), self.tokenizer, max_length=self.max_length)
        val_dataset = ToxicityDataset(X_val, y_val.tolist(), self.tokenizer, max_length=self.max_length)
        
        # Динамический паддинг до макс. длины в батче (быстрее и меньше лишних паддингов)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors='pt')
        
        def collate_fn(batch):
            labels = torch.stack([b['labels'] for b in batch])
            tokenizer_batch = data_collator([
                {'input_ids': b['input_ids'], 'attention_mask': b['attention_mask']} for b in batch
            ])
            return {**tokenizer_batch, 'labels': labels}
        
        # Длины для бакетного батчинга (round-robin по бакетам, неполные батчи отдаём)
        train_lengths = []
        for i in range(0, len(X_train), length_batch_size):
            batch_texts = X_train[i : i + length_batch_size]
            out = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
            )
            train_lengths.extend(out['length'])
        bucket_boundaries = get_bucket_boundaries(self.max_length)
        train_sampler = BucketBatchSampler(
            lengths=train_lengths,
            bucket_boundaries=bucket_boundaries,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            seed=self.random_state,
        )
        
        # DataLoader-ы (train — с бакетами, val — без)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Подготовка alpha для focal loss
        alpha = self.focal_alpha if self.focal_alpha is not None else compute_auto_alpha(y_train)

        if self.no_freeze:
            # Создаём параметры с разными скоростями обучения
            embedding_params = []
            encoder_params = []
            classifier_params = []

            for name, param in self.model.named_parameters():
                if "embeddings" in name:
                    embedding_params.append(param)
                elif "classifier" in name:
                    classifier_params.append(param)
                else:
                    encoder_params.append(param)

            # Оптимизатор с разными learning rates
            optimizer = torch.optim.AdamW([
                {'params': embedding_params, 'lr': 1e-5},      # Эмбеддинги: медленнее
                {'params': encoder_params, 'lr': 2e-5},        # Энкодер: стандартная скорость
                {'params': classifier_params, 'lr': self.learning_rate},     # Голова: быстрее
            ], weight_decay=self.weight_decay)
            
        else:
            # Оптимизатор только для незамороженных слоев
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        # Scheduler по шагам обучения
        total_steps = len(train_loader) * self.epochs
        if self.warmup_steps and self.warmup_steps > 0:
            num_warmup_steps = int(self.warmup_steps)
        elif self.warmup_ratio and self.warmup_ratio > 0:
            num_warmup_steps = int(total_steps * float(self.warmup_ratio))
        else:
            num_warmup_steps = 0

        scheduler = None
        if total_steps > 0:
            if self.lr_scheduler_type == 'linear':
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps
                )
            elif self.lr_scheduler_type == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps
                )
            elif self.lr_scheduler_type == 'constant':
                scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps
                )        

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


        # Явный цикл обучения
        best_val_ap = 0.0
        patience = 3
        patience_counter = 0
        self._best_model_state = None

        print("\nНачинаем обучение...")

        for epoch in range(self.epochs):
            train_sampler.set_epoch(epoch)
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            # TRAIN
            self.model.train()
            train_loss = 0.0
            train_probs = []
            train_labels = []

            for batch in tqdm(train_loader, desc='Обучение'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits  # (batch, 1)
                loss = criterion(logits, labels.view(-1, 1))
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_loss += loss.item()
                batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                train_probs.extend(batch_probs.tolist())
                train_labels.extend(labels.detach().cpu().numpy().tolist())

            train_loss /= max(1, len(train_loader))
            train_ap = average_precision_score(train_labels, train_probs) if len(set(train_labels)) > 1 else 0.0

            # VAL
            self.model.eval()
            val_loss = 0.0
            val_probs = []
            val_labels = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Валидация'):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    loss = criterion(logits, labels.view(-1, 1))

                    val_loss += loss.item()
                    batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                    val_probs.extend(batch_probs.tolist())
                    val_labels.extend(labels.detach().cpu().numpy().tolist())

            val_loss /= max(1, len(val_loader))
            if len(set(val_labels)) > 1:
                val_ap = average_precision_score(val_labels, val_probs)
                val_preds_05 = (np.array(val_probs) >= 0.5).astype(int)
                val_precision = precision_score(val_labels, val_preds_05, zero_division=0)
                val_recall = recall_score(val_labels, val_preds_05, zero_division=0)
                val_f1 = f1_score(val_labels, val_preds_05, zero_division=0)
            else:
                val_ap = 0.0
                val_precision = val_recall = val_f1 = 0.0

            print(f"  Train Loss: {train_loss:.4f}, Train AP: {train_ap:.4f}")
            print(f"  Val   Loss: {val_loss:.4f}, Val   AP: {val_ap:.4f}")
            print(f"  Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f} (threshold=0.5)")

            # Early stopping по лучшему AP
            if self._best_model_state is None or val_ap > best_val_ap:
                best_val_ap = val_ap
                self.best_score = val_ap
                self._best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print(f"  ✓ Новая лучшая модель (AP: {val_ap:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping после {epoch + 1} эпох")
                    break

        # Загружаем лучшую модель перед подбором порога
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)

        # Финальный прогон по валидации для подбора оптимального порога
        self.model.eval()
        val_probs_final = []
        val_labels_final = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                val_probs_final.extend(batch_probs.tolist())
                val_labels_final.extend(labels.detach().cpu().numpy().tolist())

        val_probs_final = np.array(val_probs_final)
        val_labels_final = np.array(val_labels_final)

        optimal_threshold, opt_precision, opt_recall, opt_f1 = find_threshold_max_f1_min_precision(
            val_labels_final, val_probs_final, min_precision=0.9
        )

        val_preds_optimal = (val_probs_final >= optimal_threshold).astype(int)
        # opt_f1 = f1_score(val_labels_final, val_preds_optimal, zero_division=0)

        print(f"\nОбучение завершено. Лучший Val AP: {best_val_ap:.4f}")
        print(f"\nОптимальный порог: {optimal_threshold:.4f}")
        print(f"  Precision: {opt_precision:.4f}, Recall: {opt_recall:.4f}, F1: {opt_f1:.4f}")

        self.optimal_threshold = float(optimal_threshold)

        # Сохраняем модель и токенизатор в формате HuggingFace
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"\nМодель сохранена в {output_dir}")
    
    def save_params(self, params_path: str):
        """Сохраняет параметры модели в JSON"""
        metadata = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'warmup_ratio': self.warmup_ratio,
            'lr_scheduler_type': self.lr_scheduler_type,
            'freeze_encoder': self.freeze_encoder,
            'freeze_last_n_layers': self.freeze_last_n_layers,
            'dropout': self.dropout,
            'best_score': self.best_score,
            'best_score_metric': 'average_precision',
            'optimal_threshold': self.optimal_threshold,
            'random_state': self.random_state,
            'loss_type': self.loss_type,
            'focal_gamma': self.focal_gamma,
            'focal_alpha': self.focal_alpha,
            'focal_auto_alpha': self.focal_auto_alpha,
        }
        
        # Преобразуем numpy типы
        metadata = convert_to_json_serializable(metadata)
        
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Параметры сохранены: {params_path}")


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Дообучение BERT модели для классификации токсичности')
    add_common_data_args(parser)
    add_common_output_arg(parser, default_output_dir='models/toxicity/bert')
    parser.add_argument(
        '--model-name',
        type=str,
        default='cointegrated/rubert-tiny2',
        help='Название модели из HuggingFace'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=1024,
        help='Максимальная длина последовательности'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Размер батча'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Количество эпох'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=0,
        help='Количество шагов warmup (приоритет над --warmup-ratio)'
    )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0,
        help='Доля шагов для warmup'
    )
    parser.add_argument(
        '--lr-scheduler-type',
        type=str,
        choices=['linear', 'cosine', 'constant'],
        default='linear',
        help='Тип learning rate scheduler'
    )
    parser.add_argument(
        '--no-freeze',
        action='store_true',
        help='Не замораживать модель'
    )
    parser.set_defaults(no_freeze=False)
    parser.add_argument(
        '--freeze-encoder',
        action='store_true',
        default=True,
        help='Заморозить энкодер (кроме последнего слоя)'
    )
    parser.add_argument(
        '--no-freeze-encoder',
        action='store_false',
        dest='freeze_encoder',
        help='Не замораживать энкодер'
    )
    parser.set_defaults(freeze_encoder=True)
    parser.add_argument(
        '--freeze-last-n-layers',
        type=int,
        default=0,
        help='Количество последних слоев для заморозки (0 = разморозить только последний слой)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout для классификационной головы'
    )
    add_common_random_state_arg(parser)
    add_common_loss_args(parser)
    parser.set_defaults(focal_auto_alpha=True)
    parser.add_argument(
        '--quantize-onnx',
        action='store_true',
        help='После обучения экспортировать модель в ONNX с квантизацией',
    )
    parser.add_argument(
        '--quantize-device',
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help='Устройство для ONNX квантизации: cpu (AVX2) или gpu (TensorRT)',
    )
    parser.add_argument(
        '--quantize-output-dir',
        type=str,
        default=None,
        help='Директория для ONNX модели (по умолчанию: <output_dir>_onnx_<device>)',
    )
    
    args = parser.parse_args()
    
    df_train, df_val, _ = load_train_val_data(
        data_path=args.data,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )
    
    # Создание trainer
    trainer = BERTModelTrainer(
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        no_freeze=args.no_freeze,
        freeze_encoder=args.freeze_encoder,
        freeze_last_n_layers=args.freeze_last_n_layers,
        dropout=args.dropout,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha = args.focal_alpha,
        focal_auto_alpha=args.focal_auto_alpha,
        random_state=args.random_state
    )
    
    # Подготовка данных
    X_train, y_train = trainer.prepare_data(df_train)
    
    if df_val is not None:
        X_val, y_val = trainer.prepare_data(df_val)
    else:
        X_val, y_val = None, None
    
    # Обучение модели
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.train(
        X_train, 
        y_train, 
        X_val=X_val, 
        y_val=y_val,
        output_dir=str(output_dir)
    )
    
    # Сохранение параметров
    trainer.save_params(str(output_dir / 'params.json'))

    # Опциональная ONNX квантизация
    if args.quantize_onnx:
        if not _HAS_QUANTIZE_ONNX:
            print("\nВнимание: optimum[onnxruntime] не установлен. Пропуск квантизации.")
            print("Установите: pip install optimum[onnxruntime]")
        else:
            quantize_dir = args.quantize_output_dir or f"{output_dir}_onnx_{args.quantize_device}"
            print(f"\nONNX квантизация ({args.quantize_device}) -> {quantize_dir}")
            quantize_bert_to_onnx(
                str(output_dir),
                quantize_dir,
                device=args.quantize_device,
                calibration_texts=X_train[:200] if args.quantize_device == 'gpu' else None,
            )
    
    print("\nДообучение завершено!")


if __name__ == '__main__':
    main()

