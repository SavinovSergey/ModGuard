"""Менеджер моделей с поддержкой fallback"""
import time
import logging
from typing import Dict, Optional, List

from app.models.base import BaseToxicityModel
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Управляет загрузкой и переключением моделей с поддержкой fallback"""
    
    def __init__(self, fallback_chain: Optional[List[str]] = None):
        """
        Args:
            fallback_chain: Цепочка моделей для fallback (по умолчанию из конфига)
        """
        self.models: Dict[str, BaseToxicityModel] = {}
        self.current_model: Optional[BaseToxicityModel] = None
        self.fallback_chain = fallback_chain or settings.fallback_chain
        self.model_timeouts = settings.model_timeouts
        self.model_stats: Dict[str, Dict[str, int]] = {}
    
    def register_model(self, name: str, model: BaseToxicityModel):
        """
        Регистрирует модель в менеджере
        
        Args:
            name: Имя модели
            model: Экземпляр модели
        """
        self.models[name] = model
        self.model_stats[name] = {
            "errors": 0,
            "timeouts": 0,
            "success": 0
        }
        logger.info(f"Registered model: {name}")
    
    def set_current_model(self, name: str):
        """
        Устанавливает текущую модель по имени
        
        Args:
            name: Имя модели
            model_path: Путь к файлу модели (опционально)
        
        Raises:
            ValueError: Если модель не зарегистрирована
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")
        
        try:
            self.current_model = self.models[name]
            logger.info(f"Loaded model: {name}")
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            raise
    
    def get_current_model(self) -> BaseToxicityModel:
        """
        Возвращает текущую активную модель
        
        Returns:
            Текущая модель
        
        Raises:
            RuntimeError: Если модель не загружена
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded")
        return self.current_model
    
    def get_model_with_fallback(
        self, 
        preferred_model: Optional[str] = None
    ) -> BaseToxicityModel:
        """
        Получает модель с автоматическим fallback при проблемах
        
        Args:
            preferred_model: Предпочтительная модель (если None, используется первая в цепочке)
        
        Returns:
            Рабочая модель из цепочки fallback
        
        Raises:
            RuntimeError: Если нет доступных моделей
        """
        chain = [preferred_model] if preferred_model else []
        chain.extend([m for m in self.fallback_chain if m != preferred_model])
        
        for model_name in chain:
            if model_name not in self.models:
                continue
            
            model = self.models[model_name]
            stats = self.model_stats[model_name]
            
            # Проверяем статистику ошибок
            if stats["errors"] > 10 and stats["success"] > 0:
                error_rate = stats["errors"] / (stats["errors"] + stats["success"])
                if error_rate > 0.5:  # Если >50% ошибок, пропускаем
                    logger.warning(
                        f"Model {model_name} has high error rate ({error_rate:.2%}), skipping"
                    )
                    continue
            
            if not model.is_loaded:
                try:
                    model.load()
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    stats["errors"] += 1
                    continue
            
            return model
        
        raise RuntimeError("No working model available in fallback chain")
    
    def predict_with_fallback(
        self, 
        text: str, 
        preferred_model: Optional[str] = None
    ) -> Dict:
        """
        Предсказывает с автоматическим fallback при таймауте или ошибке
        
        Args:
            text: Текст для классификации
            preferred_model: Предпочтительная модель
        
        Returns:
            Результат классификации
        
        Raises:
            RuntimeError: Если все модели в цепочке fallback не сработали
        """
        chain = [preferred_model] if preferred_model else []
        chain.extend([m for m in self.fallback_chain if m != preferred_model])
        
        last_error = None
        
        for model_name in chain:
            if model_name not in self.models:
                continue
            
            model = self.models[model_name]
            stats = self.model_stats[model_name]
            timeout = self.model_timeouts.get(model_name, 0.2)
            
            try:
                start_time = time.time()
                result = model.predict(text)
                elapsed = time.time() - start_time
                logger.info(model_name)
                if elapsed > timeout:
                    logger.warning(
                        f"Model {model_name} exceeded timeout "
                        f"({elapsed:.3f}s > {timeout}s), falling back to next model"
                    )
                    stats["timeouts"] += 1
                    continue
                
                # Успешное предсказание
                stats["success"] += 1
                logger.debug(
                    f"Successfully predicted with {model_name} in {elapsed:.3f}s"
                )
                return result
                
            except Exception as e:
                logger.error(f"Error in model {model_name}: {e}")
                stats["errors"] += 1
                last_error = e
                continue
        
        # Если все модели не сработали
        raise RuntimeError(
            f"All models in fallback chain failed. Last error: {last_error}"
        )
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Возвращает статистику использования моделей"""
        return self.model_stats.copy()




