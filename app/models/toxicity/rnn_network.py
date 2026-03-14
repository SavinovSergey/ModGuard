"""Архитектура нейронной сети RNN для классификации токсичности"""
import torch
import torch.nn as nn
from typing import Literal

RNNType = Literal['rnn', 'gru', 'lstm']


class RNNClassifier(nn.Module):
    """RNN классификатор для бинарной классификации токсичности"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 128,
        num_layers: int = 1,
        rnn_type: RNNType = 'gru',
        dropout: float = 0.2,
        bidirectional: bool = False,
        padding_idx: int = 0,
        embedding_dropout: float = 0.0,
        use_layer_norm: bool = False
    ):
        """
        Args:
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            hidden_size: Размерность скрытого состояния RNN
            num_layers: Количество слоев RNN
            rnn_type: Тип RNN ('rnn', 'gru', 'lstm')
            dropout: Dropout для RNN слоев
            bidirectional: Использовать ли двунаправленную RNN
            padding_idx: Индекс токена padding для embedding слоя
            embedding_dropout: Dropout для embedding слоя (0.0 = отключен)
            use_layer_norm: Использовать ли Layer Normalization после RNN
        """
        super(RNNClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        self.embedding_dropout = embedding_dropout
        self.use_layer_norm = use_layer_norm
        
        # Embedding слой
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # Dropout для embedding (если включен)
        if embedding_dropout > 0.0:
            self.embedding_dropout_layer = nn.Dropout(embedding_dropout)
        else:
            self.embedding_dropout_layer = None

        rnn_params = {
            'input_size': embedding_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout if num_layers > 1 else 0,
            'bidirectional': bidirectional,
            'batch_first': True
        }
        
        # RNN слой
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(**rnn_params)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(**rnn_params)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(**rnn_params)
        else:
            raise ValueError(f"Неизвестный тип RNN: {rnn_type}. Используйте 'rnn', 'gru' или 'lstm'")
        
        # Размерность выхода RNN (учитываем bidirectional)
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Layer Normalization после RNN (если включена)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(rnn_output_size)
        else:
            self.layer_norm = None
        
        # Промежуточный полносвязный слой с нелинейностью
        self.fc1 = nn.Linear(rnn_output_size, rnn_output_size // 2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # Финальный полносвязный слой для классификации
        self.fc2 = nn.Linear(rnn_output_size // 2, 1)

        # Dropout для регуляризации
        self.dropout_layer = nn.Dropout(dropout)
    
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Прямой проход через сеть
        
        Args:
            x: Тензор входных данных [batch_size, seq_length]
            lengths: Длины последовательностей для packed sequence (опционально)
        
        Returns:
            Логиты для бинарной классификации [batch_size, 1]
        """
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # Если эмбеддинги в FP16, преобразуем в float32 для совместимости с RNN слоями
        # RNN слои обычно работают с float32, а не float16
        if embedded.dtype == torch.float16:
            embedded = embedded.float()
        
        # Применяем dropout на эмбеддингах если включен
        if self.embedding_dropout_layer is not None:
            embedded = self.embedding_dropout_layer(embedded)
        
        # RNN
        lengths_cpu = None
        if lengths is not None:
            lengths_cpu = lengths.detach().cpu()
            # Используем packed sequence для эффективной обработки переменной длины
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
        
        rnn_output, _ = self.rnn(embedded)
        
        # Распаковываем последовательность если использовали packed sequence
        if lengths is not None:
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )
            batch_size = rnn_output.size(0)
            # Для индексации приводим длины к устройству выхода RNN.
            last_indices = (lengths_cpu.to(rnn_output.device) - 1).long()
            batch_indices = torch.arange(batch_size, device=rnn_output.device)

            if self.bidirectional:
                # Для bidirectional корректная агрегация:
                # - forward: последний валидный timestep
                # - backward: первый timestep (там аккумулирована информация о всей последовательности)
                forward_last = rnn_output[batch_indices, last_indices, :self.hidden_size]
                backward_first = rnn_output[batch_indices, 0, self.hidden_size:]
                final_hidden = torch.cat([forward_last, backward_first], dim=1)
            else:
                final_hidden = rnn_output[batch_indices, last_indices, :]
        else:
            if self.bidirectional:
                forward_last = rnn_output[:, -1, :self.hidden_size]
                backward_first = rnn_output[:, 0, self.hidden_size:]
                final_hidden = torch.cat([forward_last, backward_first], dim=1)
            else:
                # Если не использовали packed sequence, берем последний выход
                final_hidden = rnn_output[:, -1, :]
        
        # Layer Normalization после RNN (если включена)
        if self.layer_norm is not None:
            final_hidden = self.layer_norm(final_hidden)
        
        # Промежуточный полносвязный слой с нелинейностью
        final_hidden = self.fc1(final_hidden)
        final_hidden = self.tanh(final_hidden)
        final_hidden = self.dropout_layer(final_hidden)
        
        # Финальный полносвязный слой
        output = self.fc2(final_hidden)  # [batch_size, 1]
        
        return output
    
    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Предсказывает вероятности
        
        Args:
            x: Тензор входных данных [batch_size, seq_length]
            lengths: Длины последовательностей (опционально)
        
        Returns:
            Вероятности токсичности [batch_size, 1]
        """
        with torch.no_grad():
            logits = self.forward(x, lengths)
            probs = torch.sigmoid(logits)
        return probs


