"""FastText модель для классификации токсичности"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import fasttext

from app.models.base import ClassicalTextModelBase
import logging

logger = logging.getLogger(__name__)


class FastTextModel(ClassicalTextModelBase):
    """Модель классификации токсичности на основе FastText"""
    
    def __init__(
        self, 
        model_path: Optional[str] = None
    ):
        """
        Args:
            model_path: Путь к сохраненной FastText модели
        """
        super().__init__(model_name="fasttext", model_type="fasttext")
        self.model_path = model_path
        self.model: Optional[fasttext.FastText] = None
        self.optimal_threshold: float = 0.5  # Порог по умолчанию
    
    def load(self, model_path: Optional[str] = None, params_path: Optional[str] = None) -> None:
        """
        Загружает FastText модель из файла
        
        Args:
            model_path: Путь к модели (если не указан, используется self.model_path)
            params_path: Путь к файлу параметров (params.json) для загрузки optimal_threshold
        """
        model_path = model_path or self.model_path
        
        if model_path is None:
            raise ValueError(
                "Необходимо указать путь к модели. "
                "Используйте load(model_path='...')"
            )
        
        # Проверяем существование файла
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        # Загружаем модель
        try:
            self.model = fasttext.load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели: {e}")
        
        # Загружаем optimal_threshold из params.json, если доступен
        if params_path is None:
            # Пытаемся найти params.json в той же директории, что и модель
            model_dir = Path(model_path).parent
            params_path = model_dir / 'params.json'
        
        if Path(params_path).exists():
            try:
                with open(params_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                    if 'optimal_threshold' in params:
                        self.optimal_threshold = float(params['optimal_threshold'])
                        logger.info(f"Загружен optimal_threshold: {self.optimal_threshold}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить optimal_threshold из {params_path}: {e}")
                logger.info(f"Используется порог по умолчанию: {self.optimal_threshold}")
        else:
            logger.info(f"Файл params.json не найден, используется порог по умолчанию: {self.optimal_threshold}")
        
        # Сохраняем путь для будущего использования
        self.model_path = model_path
        
        self.is_loaded = True
        logger.info(f"FastText модель загружена из {model_path}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Предсказывает токсичность для одного текста
        
        Args:
            text: Текст для классификации
        
        Returns:
            Словарь с результатами классификации
        """
        self.ensure_loaded()
        
        # Предобработка текста
        processed_text = self.preprocess_text(text)
        
        if not processed_text or not processed_text.strip():
            return self.empty_result()
        
        # Предсказание через FastText
        labels, probas = self.model.predict(processed_text)

        # Определяем вероятность токсичности
        is_toxic_label = labels[0].endswith('1')
        toxicity_score = float(probas[0] if is_toxic_label else 1 - probas[0])
        # Используем optimal_threshold вместо дефолтного порога
        is_toxic = toxicity_score >= self.optimal_threshold
        
        return {
            'is_toxic': is_toxic,
            'toxicity_score': float(toxicity_score),
            'toxicity_types': {}  # FastText модель не определяет типы токсичности
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Предсказывает токсичность для батча текстов
        
        Args:
            texts: Список текстов для классификации
        
        Returns:
            Список словарей с результатами классификации
        """
        self.ensure_loaded()
        
        if not texts:
            return []
        
        # Предобработка текстов
        processed_texts = self.preprocess_batch(texts)
        
        # Определяем индексы непустых текстов
        non_empty_indices = self.non_empty_indexes(processed_texts)
        non_empty_texts = [processed_texts[i] for i in non_empty_indices]
        
        # Если все тексты пустые, возвращаем нулевые вероятности
        if not non_empty_texts:
            return [self.empty_result()] * len(texts)
        
        # Предсказание через FastText только для непустых текстов
        labels, probas = self.model.predict(non_empty_texts)
        y_proba = [p[0] if l[0].endswith('1') else 1 - p[0] for l, p in zip(labels, probas)]
        
        # Формируем результаты для всех текстов
        results = []
        proba_idx = 0
        for i, text in enumerate(texts):
            if i in non_empty_indices:
                # Текст непустой, используем результат модели
                toxicity_score = float(y_proba[proba_idx])
                # Используем optimal_threshold вместо дефолтного порога
                is_toxic = toxicity_score >= self.optimal_threshold
                results.append({
                    'is_toxic': is_toxic,
                    'toxicity_score': toxicity_score,
                    'toxicity_types': {}
                })
                proba_idx += 1
            else:
                # Текст пустой, возвращаем нулевую вероятность
                results.append({
                    'is_toxic': False,
                    'toxicity_score': 0.0,
                    'toxicity_types': {}
                })

        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели"""
        info = self._base_info("FastText supervised classification model for toxicity detection")
        
        if self.is_loaded and self.model is not None:
            try:
                # Получаем параметры модели
                info['model_params'] = {
                    'dim': getattr(self.model, 'dim', None),
                    'epoch': getattr(self.model, 'epoch', None),
                    'lr': getattr(self.model, 'lr', None),
                    'word_ngrams': getattr(self.model, 'wordNgrams', None),
                }
            except Exception as e:
                logger.warning(f"Не удалось получить параметры модели: {e}")
                info['model_params'] = {}
        
        return info