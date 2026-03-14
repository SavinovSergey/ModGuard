"""TF-IDF модель для классификации токсичности"""
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.models.base import ClassicalTextModelBase
import logging


class TfidfModel(ClassicalTextModelBase):
    """Модель классификации токсичности на основе TF-IDF и логистической регрессии"""
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        vectorizer_path: Optional[str] = None
    ):
        """
        Args:
            model_path: Путь к сохраненной модели
            vectorizer_path: Путь к сохраненному векторизатору
        """
        super().__init__(model_name="tfidf", model_type="tfidf")
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.optimal_threshold: float = 0.5  # Порог по умолчанию

    
    def load(
        self, 
        model_path: Optional[str] = None, 
        vectorizer_path: Optional[str] = None,
        params_path: Optional[str] = None
    ) -> None:
        """
        Загружает модель и векторизатор из файлов
        
        Args:
            model_path: Путь к модели (если не указан, используется self.model_path)
            vectorizer_path: Путь к векторизатору (если не указан, используется self.vectorizer_path)
            params_path: Путь к файлу параметров (params.json) для загрузки optimal_threshold
        """
        model_path = model_path or self.model_path
        vectorizer_path = vectorizer_path or self.vectorizer_path
        

        if model_path is None or vectorizer_path is None:
            raise ValueError(
                "Необходимо указать пути к модели и векторизатору. "
                "Используйте load(model_path='...', vectorizer_path='...')"
            )
        
        # Проверяем существование файлов
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if not Path(vectorizer_path).exists():
            raise FileNotFoundError(f"Файл векторизатора не найден: {vectorizer_path}")
        
        # Загружаем модель и векторизатор
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
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
                        logging.info(f"Загружен optimal_threshold: {self.optimal_threshold}")
            except Exception as e:
                logging.warning(f"Не удалось загрузить optimal_threshold из {params_path}: {e}")
                logging.info(f"Используется порог по умолчанию: {self.optimal_threshold}")
        else:
            logging.info(f"Файл params.json не найден, используется порог по умолчанию: {self.optimal_threshold}")
        
        # Сохраняем пути для будущего использования
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        self.is_loaded = True
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Предсказывает токсичность для одного текста
        
        Args:
            text: Предобработанный текст
        
        Returns:
            Словарь с результатами классификации
        """
        self.ensure_loaded()
        
        text = self.preprocess_text(text)

        if not text or not text.strip():
            return self.empty_result()
        
        # Векторизация
        X = self.vectorizer.transform([text])
        
        # Предсказание вероятности
        proba = self.model.predict_proba(X)[0]
        toxicity_score = float(proba[1])  # Вероятность класса 1 (токсичный)
        # Используем optimal_threshold вместо дефолтного порога
        is_toxic = toxicity_score >= self.optimal_threshold
        
        return {
            'is_toxic': is_toxic,
            'toxicity_score': toxicity_score,
            'toxicity_types': {}  # TF-IDF модель не определяет типы токсичности
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Предсказывает токсичность для батча текстов
        
        Args:
            texts: Список предобработанных текстов
        
        Returns:
            Список словарей с результатами классификации
        """
        self.ensure_loaded()
        
        texts = self.preprocess_batch(texts)

        if not texts:
            return []
        
        # Фильтруем пустые тексты
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            return [self.empty_result()] * len(texts)
        
        # Векторизация
        X = self.vectorizer.transform(non_empty_texts)
        
        # Предсказание вероятностей
        probabilities = self.model.predict_proba(X)
        
        # Формируем результаты
        results = []
        idx = 0
        for text in texts:
            if text and text.strip():
                proba = probabilities[idx]
                toxicity_score = float(proba[1])
                # Используем optimal_threshold вместо дефолтного порога
                is_toxic = toxicity_score >= self.optimal_threshold
                idx += 1
            else:
                toxicity_score = 0.0
                is_toxic = False
            
            results.append({
                'is_toxic': is_toxic,
                'toxicity_score': toxicity_score,
                'toxicity_types': {}
            })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели"""
        info = self._base_info("TF-IDF + Logistic Regression toxicity classification model")
        
        if self.is_loaded and self.model is not None:
            info['model_params'] = {
                'C': getattr(self.model, 'C', None),
                'penalty': getattr(self.model, 'penalty', None),
                'solver': getattr(self.model, 'solver', None),
            }
            if self.vectorizer is not None:
                info['vectorizer_params'] = {
                    'max_features': getattr(self.vectorizer, 'max_features', None),
                    'ngram_range': getattr(self.vectorizer, 'ngram_range', None),
                }
        
        return info

