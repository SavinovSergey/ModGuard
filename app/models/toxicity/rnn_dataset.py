"""PyTorch Dataset для обучения RNN модели"""
import torch
from torch.utils.data import Dataset
from typing import List, Optional

from app.models.toxicity.rnn_tokenizers import BaseRNNTokenizer


class ToxicityDataset(Dataset):
    """Датасет для классификации токсичности"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BaseRNNTokenizer,
        max_length: Optional[int] = None
    ):
        """
        Args:
            texts: Список текстов
            labels: Список меток (0 или 1)
            tokenizer: Токенизатор для кодирования текстов
            max_length: Максимальная длина последовательности
        """
        if len(texts) != len(labels):
            raise ValueError(f"Количество текстов ({len(texts)}) не совпадает с количеством меток ({len(labels)})")
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Возвращает размер датасета"""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Возвращает один элемент датасета
        
        Args:
            idx: Индекс элемента
        
        Returns:
            Словарь с токенизированным текстом и меткой
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Кодируем текст
        token_ids = self.tokenizer.encode(text, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


def collate_fn(batch: List[dict], pad_token_id: int = 0) -> dict:
    """
    Функция для батчинга данных
    
    Args:
        batch: Список элементов из датасета
    
    Returns:
        Словарь с батчированными данными
    """
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Padding до максимальной длины в батче
    max_len = max(len(ids) for ids in input_ids)
    
    # Создаем тензоры с padding
    padded_input_ids = []
    lengths = []
    
    for ids in input_ids:
        # Реальная длина последовательности = количество токенов, отличных от PAD.
        # Это важно для pack_padded_sequence.
        non_pad_length = int((ids != pad_token_id).sum().item())
        length = max(1, non_pad_length)
        lengths.append(length)
        
        if len(ids) < max_len:
            # Добавляем padding с корректным индексом pad-токена
            padding = torch.full((max_len - len(ids),), pad_token_id, dtype=torch.long)
            padded = torch.cat([ids, padding])
        else:
            padded = ids
        
        padded_input_ids.append(padded)
    
    # Стекируем в один тензор
    input_ids_tensor = torch.stack(padded_input_ids)
    labels_tensor = torch.stack(labels)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return {
        'input_ids': input_ids_tensor,
        'labels': labels_tensor,
        'lengths': lengths_tensor
    }



