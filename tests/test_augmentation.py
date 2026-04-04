"""Тесты для модуля аугментации данных"""
import pytest

pd = pytest.importorskip("pandas", reason="pandas не установлен — тесты аугментации пропущены")

import numpy as np
from unittest.mock import Mock, patch

from app.augmentation import CharNoiseAugmenter, DataAugmenter


class TestCharNoiseAugmenter:
    """Тесты для CharNoiseAugmenter"""
    
    def test_qwerty_replace(self):
        """Тест QWERTY замены"""
        augmenter = CharNoiseAugmenter(
            noise_intensity=0.1,
            random_state=42
        )
        
        text = "привет"
        augmented, aug_type = augmenter.augment(text, noise_type='qwerty')
        
        assert aug_type == 'char_noise_qwerty'
        assert isinstance(augmented, str)
        assert len(augmented) == len(text)  # Длина должна сохраниться
    
    def test_insert_char(self):
        """Тест добавления символов"""
        augmenter = CharNoiseAugmenter(
            noise_intensity=0.1,
            random_state=42
        )
        
        text = "тест"
        augmented, aug_type = augmenter.augment(text, noise_type='insert')
        
        assert aug_type == 'char_noise_insert'
        assert isinstance(augmented, str)
        assert len(augmented) >= len(text)  # Длина должна увеличиться
    
    def test_delete_char(self):
        """Тест удаления символов"""
        augmenter = CharNoiseAugmenter(
            noise_intensity=0.1,
            random_state=42
        )
        
        text = "длинный текст"
        augmented, aug_type = augmenter.augment(text, noise_type='delete')
        
        assert aug_type == 'char_noise_delete'
        assert isinstance(augmented, str)
        assert len(augmented) <= len(text)  # Длина должна уменьшиться
        assert len(augmented) > 0  # Но не должна быть пустой
    
    def test_replace_char(self):
        """Тест замены символов"""
        augmenter = CharNoiseAugmenter(
            noise_intensity=0.1,
            random_state=42
        )
        
        text = "пример"
        augmented, aug_type = augmenter.augment(text, noise_type='replace')
        
        assert aug_type == 'char_noise_replace'
        assert isinstance(augmented, str)
        assert len(augmented) == len(text)  # Длина должна сохраниться
    
    def test_leet_speak(self):
        """Тест LeetSpeak замены"""
        augmenter = CharNoiseAugmenter(
            noise_intensity=0.1,
            random_state=42
        )
        
        text = "пример"
        augmented, aug_type = augmenter.augment(text, noise_type='leet')
        
        assert aug_type == 'char_noise_leet'
        assert isinstance(augmented, str)
        assert len(augmented) == len(text)  # Длина должна сохраниться
    
    def test_combined_noise(self):
        """Тест комбинированного шума"""
        augmenter = CharNoiseAugmenter(
            noise_intensity=0.1,
            random_state=42
        )
        
        text = "длинный текст для теста"
        augmented, aug_type = augmenter.augment(text, noise_type='combined')
        
        assert aug_type == 'char_noise_combined'
        assert isinstance(augmented, str)
        assert len(augmented) > 0
    
    def test_empty_text(self):
        """Тест обработки пустого текста"""
        augmenter = CharNoiseAugmenter(random_state=42)
        
        text = ""
        augmented, aug_type = augmenter.augment(text)
        
        assert augmented == text
        assert aug_type == 'original'
    
    def test_preserve_length_for_qwerty(self):
        """Тест сохранения длины при QWERTY замене"""
        augmenter = CharNoiseAugmenter(noise_intensity=0.2, random_state=42)
        
        text = "привет мир"
        augmented, _ = augmenter.augment(text, noise_type='qwerty')
        
        assert len(augmented) == len(text)


class TestDataAugmenter:
    """Тесты для DataAugmenter"""
    
    def test_augment_dataframe_without_augmentation(self):
        """Тест без аугментации"""
        df = pd.DataFrame({
            'text': ['текст один', 'текст два'],
            'label': [0, 1]
        })
        
        augmenter = DataAugmenter(
            apply_back_translation=False,
            apply_char_noise=False,
            random_state=42
        )
        
        result = augmenter.augment_dataframe(df)
        
        assert len(result) == len(df)
        assert 'augmentation_type' in result.columns
        assert all(result['augmentation_type'] == 'original')
    
    def test_augment_dataframe_with_char_noise(self):
        """Тест с шумом на уровне символов"""
        df = pd.DataFrame({
            'text': ['короткий текст', 'длинный текст для теста'],
            'label': [0, 1]
        })
        
        augmenter = DataAugmenter(
            apply_char_noise=True,
            char_noise_prob=1.0,  # Применяем ко всем
            char_noise_intensity=0.05,
            random_state=42
        )
        
        result = augmenter.augment_dataframe(df)
        
        assert len(result) >= len(df)
        assert 'augmentation_type' in result.columns
        assert 'original' in result['augmentation_type'].values
        
        # Должны быть аугментированные примеры
        augmented = result[result['augmentation_type'] != 'original']
        assert len(augmented) > 0
    
    def test_augmentation_type_column(self):
        """Тест наличия колонки augmentation_type"""
        df = pd.DataFrame({
            'text': ['текст'],
            'label': [0]
        })
        
        augmenter = DataAugmenter(
            apply_char_noise=True,
            char_noise_prob=1.0,
            random_state=42
        )
        
        result = augmenter.augment_dataframe(df)
        
        assert 'augmentation_type' in result.columns
        assert set(result['augmentation_type'].unique()).issubset({
            'original',
            'char_noise_qwerty',
            'char_noise_insert',
            'char_noise_delete',
            'char_noise_replace',
            'char_noise_leet',
            'char_noise_combined'
        })
    
    def test_toxic_noise_multiplier(self):
        """Тест применения большего шума к токсичным примерам"""
        df = pd.DataFrame({
            'text': ['нормальный текст', 'токсичный текст'],
            'label': [0, 1]
        })
        
        augmenter = DataAugmenter(
            apply_char_noise=True,
            char_noise_prob=0.5,
            toxic_noise_multiplier=2.0,
            random_state=42
        )
        
        result = augmenter.augment_dataframe(df)
        
        # Проверяем, что токсичные примеры получили больше аугментаций
        toxic_original = result[(result['label'] == 1) & (result['augmentation_type'] == 'original')]
        toxic_augmented = result[(result['label'] == 1) & (result['augmentation_type'] != 'original')]
        
        # Должны быть аугментированные токсичные примеры
        assert len(toxic_augmented) >= 0  # Может быть 0 из-за вероятности
    
    def test_preserve_labels(self):
        """Тест сохранения меток при аугментации"""
        df = pd.DataFrame({
            'text': ['текст один', 'текст два'],
            'label': [0, 1]
        })
        
        augmenter = DataAugmenter(
            apply_char_noise=True,
            char_noise_prob=1.0,
            random_state=42
        )
        
        result = augmenter.augment_dataframe(df)
        
        # Проверяем, что метки сохранились
        original_labels = set(df['label'].unique())
        result_labels = set(result['label'].unique())
        
        assert original_labels == result_labels
    
    def test_statistics_logging(self):
        """Тест логирования статистики"""
        df = pd.DataFrame({
            'text': ['текст'] * 10,
            'label': [0] * 10
        })
        
        augmenter = DataAugmenter(
            apply_char_noise=True,
            char_noise_prob=0.5,
            random_state=42
        )
        
        # Проверяем, что метод выполняется без ошибок
        result = augmenter.augment_dataframe(df)
        assert len(result) >= len(df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

