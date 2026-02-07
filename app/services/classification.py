"""Сервис классификации токсичности"""
import logging
from typing import List, Dict, Any, Optional

from app.core.model_manager import ModelManager
from app.preprocessing.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class ClassificationService:
    """Сервис для классификации токсичности комментариев"""
    
    def __init__(self, model_manager: ModelManager):
        """
        Args:
            model_manager: Менеджер моделей
        """
        self.model_manager = model_manager
        self.text_processor = TextProcessor()
    
    def classify(
        self, 
        text: str, 
        context: Optional[List[str]] = None,
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Классифицирует один комментарий
        
        Args:
            text: Текст комментария
            context: Контекст обсуждения (опционально, пока не используется)
            preferred_model: Предпочтительная модель для использования
        
        Returns:
            Результат классификации
        """
        if not text:
            return {
                'is_toxic': False,
                'toxicity_score': 0.0,
                'toxicity_types': {}
            }
        
        # Получаем модель с fallback
        try:
            result = self.model_manager.predict_with_fallback(
                text, 
                preferred_model
            )
            return result
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Возвращаем безопасный результат при ошибке
            return {
                'is_toxic': False,
                'toxicity_score': 0.0,
                'toxicity_types': {},
                'error': str(e)
            }
    
    def classify_batch(
        self, 
        texts: List[str],
        preferred_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Классифицирует батч комментариев
        
        Args:
            texts: Список текстов комментариев
            preferred_model: Предпочтительная модель
        
        Returns:
            Список результатов классификации
        """
        if not texts:
            return []
        
        # Получаем модель
        model = self.model_manager.get_model_with_fallback(preferred_model)
        
        try:
            # Используем batch-метод модели, если доступен
            results = model.predict_batch(texts)
            return results
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            # Fallback: обрабатываем по одному
            logger.warning("Falling back to individual classification")
            return [
                self.classify(text, preferred_model=preferred_model) 
                for text in texts
            ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущей модели"""
        try:
            model = self.model_manager.get_current_model()
            return model.get_model_info()
        except RuntimeError:
            return {
                'name': 'none',
                'type': 'none',
                'is_loaded': False,
                'error': 'No model loaded'
            }




