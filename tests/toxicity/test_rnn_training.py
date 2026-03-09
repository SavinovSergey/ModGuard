"""Тесты для обучения RNN модели"""
import pytest
import tempfile
from pathlib import Path
import torch
import numpy as np

from app.models.toxicity.rnn_dataset import ToxicityDataset, collate_fn
from app.models.toxicity.rnn_tokenizers import BPETokenizer
from app.models.toxicity.rnn_network import RNNClassifier
from scripts.toxicity.train_rnn import RNNModelTrainer, BinaryFocalLoss
from app.preprocessing.text_processor import TextProcessor


@pytest.fixture
def sample_texts_and_labels():
    """Возвращает примеры текстов и меток для тестирования"""
    text_processor = TextProcessor(use_lemmatization=False, remove_stopwords=False)
    
    texts = [
        "это нормальный комментарий",
        "это токсичный ебать комментарий",
        "спасибо за информацию",
        "иди нахуй",
        "интересная статья",
        "ты дебил",
        "хорошая работа",
        "отличный контент"
    ]
    labels = [0, 1, 0, 1, 0, 1, 0, 0]
    
    # Предобработка
    processed_texts = [text_processor.normalize(text) for text in texts]
    
    return processed_texts, np.array(labels, dtype=np.float32)


@pytest.fixture
def trained_tokenizer(sample_texts_and_labels):
    """Создает обученный токенизатор"""
    texts, _ = sample_texts_and_labels
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(texts)
    return tokenizer


def test_collate_fn_basic(trained_tokenizer):
    """Тест базовой функциональности collate_fn"""
    pad_token_id = trained_tokenizer.get_pad_token_id()
    
    batch = [
        {
            'input_ids': torch.tensor(trained_tokenizer.encode("текст один", max_length=10)),
            'label': torch.tensor(0.0)
        },
        {
            'input_ids': torch.tensor(trained_tokenizer.encode("текст два", max_length=10)),
            'label': torch.tensor(1.0)
        }
    ]
    
    result = collate_fn(batch, pad_token_id=pad_token_id)
    
    assert 'input_ids' in result
    assert 'labels' in result
    assert 'lengths' in result
    
    assert result['input_ids'].shape[0] == 2  # batch size
    assert result['labels'].shape[0] == 2
    assert len(result['lengths']) == 2


def test_collate_fn_uses_custom_pad_token_and_lengths_by_non_pad_tokens(trained_tokenizer):
    """Тест что collate_fn использует правильный pad_token_id и считает длины по не-pad токенам"""
    pad_token_id = trained_tokenizer.get_pad_token_id()
    
    # Создаем батч с разными длинами
    text1 = "короткий"
    text2 = "это более длинный текст"
    
    ids1 = trained_tokenizer.encode(text1, max_length=20)
    ids2 = trained_tokenizer.encode(text2, max_length=20)
    
    batch = [
        {
            'input_ids': torch.tensor(ids1),
            'label': torch.tensor(0.0)
        },
        {
            'input_ids': torch.tensor(ids2),
            'label': torch.tensor(1.0)
        }
    ]
    
    result = collate_fn(batch, pad_token_id=pad_token_id)
    
    # Проверяем что длины считаются правильно (по не-pad токенам)
    length1 = int((torch.tensor(ids1) != pad_token_id).sum().item())
    length2 = int((torch.tensor(ids2) != pad_token_id).sum().item())
    
    assert result['lengths'][0] == max(1, length1)
    assert result['lengths'][1] == max(1, length2)


def test_binary_focal_loss():
    """Тест BinaryFocalLoss"""
    criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    
    logits = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    
    loss = criterion(logits, targets)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_binary_focal_loss_with_none_alpha():
    """Тест BinaryFocalLoss без alpha"""
    criterion = BinaryFocalLoss(alpha=None, gamma=2.0)
    
    logits = torch.tensor([1.0, -1.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    
    loss = criterion(logits, targets)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_single_training_step_uses_bce_and_updates_model_parameters(trained_tokenizer, sample_texts_and_labels):
    """Тест что один шаг обучения использует BCE и обновляет параметры модели"""
    texts, labels = sample_texts_and_labels
    
    # Создаем модель
    vocab_size = trained_tokenizer.get_vocab_size()
    model = RNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_size=16,
        num_layers=1,
        rnn_type='gru',
        dropout=0.1,
        bidirectional=False,
        padding_idx=trained_tokenizer.get_pad_token_id(),
        embedding_dropout=0.0,
        use_layer_norm=False
    )
    
    # Создаем датасет и DataLoader
    dataset = ToxicityDataset(texts[:4], labels[:4].tolist(), trained_tokenizer, max_length=32)
    pad_token_id = trained_tokenizer.get_pad_token_id()
    collate_with_pad = lambda batch: collate_fn(batch, pad_token_id=pad_token_id)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_with_pad
    )
    
    # Получаем начальные параметры
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()
    
    # Один шаг обучения
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    batch = next(iter(dataloader))
    input_ids = batch['input_ids']
    labels_batch = batch['labels']
    lengths = batch['lengths']
    
    optimizer.zero_grad()
    logits = model(input_ids, lengths=lengths).squeeze(-1)
    loss = criterion(logits, labels_batch)
    loss.backward()
    optimizer.step()
    
    # Проверяем что параметры изменились
    params_changed = False
    for name, param in model.named_parameters():
        if not torch.equal(param.data, initial_params[name]):
            params_changed = True
            break
    
    assert params_changed, "Параметры модели должны были измениться после шага обучения"
    assert loss.item() > 0
    assert not torch.isnan(loss)


@pytest.mark.parametrize("rnn_type", ['rnn', 'gru', 'lstm'])
def test_trainer_train_cycle_runs_and_stores_best_state(rnn_type, sample_texts_and_labels):
    """Тест что цикл обучения работает и сохраняет лучшее состояние модели"""
    texts, labels = sample_texts_and_labels
    
    # Создаем trainer с минимальными параметрами для быстрого теста
    trainer = RNNModelTrainer(
        tokenizer_type='bpe',
        rnn_type=rnn_type,
        embedding_dim=32,
        hidden_size=16,
        num_layers=1,
        dropout=0.1,
        bidirectional=False,
        max_length=32,
        vocab_size=500,
        batch_size=2,
        learning_rate=0.01,
        epochs=2,
        loss_type='bce',
        embedding_dropout=0.0,
        weight_decay=1e-4,
        use_cv=False,
        use_layer_norm=False,
        max_grad_norm=1.0,
        use_lr_schedule=False,
        device='cpu',
        random_state=42
    )
    
    # Обучаем токенизатор
    trainer.train_tokenizer(texts)
    
    # Разделяем данные
    X_train = texts[:6]
    y_train = labels[:6]
    X_val = texts[6:]
    y_val = labels[6:]
    
    # Обучаем модель
    trainer._train_single_fold(X_train, y_train, X_val, y_val, fold_idx=None)
    
    # Проверяем что лучшее состояние сохранено
    assert trainer.best_model_state is not None
    assert trainer.best_score >= 0.0
    assert trainer.model is not None


def test_trainer_gradient_clipping(sample_texts_and_labels):
    """Тест что gradient clipping работает"""
    texts, labels = sample_texts_and_labels
    
    trainer = RNNModelTrainer(
        tokenizer_type='bpe',
        rnn_type='gru',
        embedding_dim=32,
        hidden_size=16,
        max_length=32,
        vocab_size=500,
        batch_size=2,
        learning_rate=0.01,
        epochs=1,
        max_grad_norm=0.5,  # Включаем clipping
        use_lr_schedule=False,
        use_cv=False,
        device='cpu',
        random_state=42
    )
    
    trainer.train_tokenizer(texts)
    
    X_train = texts[:4]
    y_train = labels[:4]
    X_val = texts[4:]
    y_val = labels[4:]
    
    # Обучаем - не должно быть ошибок
    trainer._train_single_fold(X_train, y_train, X_val, y_val, fold_idx=None)
    
    assert trainer.best_model_state is not None


def test_trainer_layer_norm(sample_texts_and_labels):
    """Тест что Layer Normalization работает"""
    texts, labels = sample_texts_and_labels
    
    trainer = RNNModelTrainer(
        tokenizer_type='bpe',
        rnn_type='gru',
        embedding_dim=32,
        hidden_size=16,
        max_length=32,
        vocab_size=500,
        batch_size=2,
        learning_rate=0.01,
        epochs=1,
        use_layer_norm=True,  # Включаем Layer Norm
        use_lr_schedule=False,
        use_cv=False,
        device='cpu',
        random_state=42
    )
    
    trainer.train_tokenizer(texts)
    
    X_train = texts[:4]
    y_train = labels[:4]
    X_val = texts[4:]
    y_val = labels[4:]
    
    # Обучаем - не должно быть ошибок
    trainer._train_single_fold(X_train, y_train, X_val, y_val, fold_idx=None)
    
    assert trainer.best_model_state is not None
    # Проверяем что модель имеет layer_norm
    base_model = trainer._unwrap_model(trainer.model)
    assert hasattr(base_model, 'layer_norm')
    assert base_model.layer_norm is not None



