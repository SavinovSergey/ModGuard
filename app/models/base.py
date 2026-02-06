"""Базовый интерфейс для всех моделей классификации токсичности"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseToxicityModel(ABC):
    """Базовый класс для всех моделей классификации токсичности"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_loaded = False
    
    @abstractmethod
    def load(self, model_path: str = None) -> None:
        """
        Загружает модель из файла или инициализирует
        
        Args:
            model_path: Путь к файлу модели (опционально)
        """
        pass
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Предсказывает токсичность для одного текста
        
        Args:
            text: Предобработанный текст для классификации
        
        Returns:
            {
                'is_toxic': bool,
                'toxicity_score': float,  # 0-1
                'toxicity_types': Dict[str, float]  # опционально
            }
        """
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Предсказывает токсичность для батча текстов
        
        Args:
            texts: Список предобработанных текстов
        
        Returns:
            Список словарей с результатами классификации
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели
        
        Returns:
            {
                'name': str,
                'type': str,
                'is_loaded': bool,
                'version': str,
                'description': str
            }
        """
        pass



