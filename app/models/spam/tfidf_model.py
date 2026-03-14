"""TF-IDF модель для классификации спама (опционально + ручные признаки)."""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.features.spam_features import (
    matches_caps_word_double_excl_rule,
    matches_caps_word_rule,
)
from app.preprocessing.spam_processor import SpamTextProcessor

logger = logging.getLogger(__name__)

# Порог для правила «слово капсом + !!» — при срабатывании сразу спам
RULE_SPAM_SCORE = 1.0


class SpamTfidfModel:
    """
    Модель спама: TF-IDF + LogisticRegression.
    При наличии scaler.pkl и use_extra_features в params — добавляет ручные признаки (капс, URL, длина, опечатки и т.д.).
    """

    def __init__(self) -> None:
        self.model: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.vectorizer_caps: Optional[TfidfVectorizer] = None
        self.use_caps_rest_split: bool = False
        self.preprocessor = SpamTextProcessor()
        self.model_path: Optional[str] = None
        self.vectorizer_path: Optional[str] = None
        self.optimal_threshold: float = 0.5
        self.is_loaded = False
        self.use_extra_features: bool = False
        self.scaler: Optional[Any] = None

    def load(
        self,
        model_path: str,
        vectorizer_path: str,
        params_path: Optional[str] = None,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if not Path(vectorizer_path).exists():
            raise FileNotFoundError(f"Файл векторизатора не найден: {vectorizer_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        # Совместимость sklearn: старые сохранённые модели могут не иметь атрибута multi_class
        if hasattr(self.model, "__dict__") and not hasattr(self.model, "multi_class"):
            try:
                setattr(self.model, "multi_class", "ovr")
            except Exception:
                pass
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        model_dir = Path(model_path).parent
        caps_path = model_dir / "vectorizer_caps.pkl"
        if caps_path.exists():
            with open(caps_path, "rb") as f:
                self.vectorizer_caps = pickle.load(f)
            self.use_caps_rest_split = True
        if params_path is None:
            params_path = str(model_dir / "params.json")
        if Path(params_path).exists():
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                    if "optimal_threshold" in params:
                        self.optimal_threshold = float(params["optimal_threshold"])
                    self.use_extra_features = params.get("use_extra_features", False)
            except Exception as e:
                logger.warning("Не удалось загрузить params для спама: %s", e)
        if self.use_extra_features:
            scaler_path = model_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            else:
                logger.warning("use_extra_features=True, но scaler.pkl не найден — используем только TF-IDF")
                self.use_extra_features = False
        self.is_loaded = True

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {"is_spam": False, "spam_score": 0.0}

    def _build_X(self, processed_texts: List[str], raw_texts: List[str]):
        """Строит матрицу: [X_tfidf | X_features]. При use_caps_rest_split — TF-IDF = [caps | rest]."""
        if self.use_caps_rest_split and self.vectorizer_caps is not None:
            from app.preprocessing.spam_processor import split_caps_rest_batch
            caps_parts, rest_parts = split_caps_rest_batch(processed_texts)
            X_caps = self.vectorizer_caps.transform(caps_parts)
            X_rest = self.vectorizer.transform(rest_parts)
            X_tfidf = sparse_hstack([X_caps, X_rest])
        else:
            X_tfidf = self.vectorizer.transform(processed_texts)
        if not self.use_extra_features or self.scaler is None:
            return X_tfidf
        from app.features.spam_features import extract_spam_features_batch
        X_feat = extract_spam_features_batch(raw_texts)
        X_feat = self.scaler.transform(X_feat)
        X_feat_sparse = csr_matrix(X_feat.astype(np.float64))
        return sparse_hstack([X_tfidf, X_feat_sparse])

    def predict(self, text: str) -> Dict[str, Any]:
        if matches_caps_word_double_excl_rule(text) or matches_caps_word_rule(text):
            return {"is_spam": True, "spam_score": RULE_SPAM_SCORE}
        if not self.is_loaded or self.model is None or self.vectorizer is None:
            return self._empty()
        processed = self.preprocessor.process(text)
        if not processed.strip():
            return self._empty()
        X = self._build_X([processed], [text])
        proba = self.model.predict_proba(X)[0]
        spam_score = float(proba[1])
        return {"is_spam": spam_score >= self.optimal_threshold, "spam_score": spam_score}

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        rule_results = [
            matches_caps_word_double_excl_rule(t) or matches_caps_word_rule(t)
            for t in texts
        ]
        out: List[Dict[str, Any]] = []
        for i, text in enumerate(texts):
            if rule_results[i]:
                out.append({"is_spam": True, "spam_score": RULE_SPAM_SCORE})
            else:
                out.append(None)
        if not self.is_loaded or not texts:
            return [o if o is not None else self._empty() for o in out]
        processed = self.preprocessor.process_batch(texts)
        non_empty_processed = []
        non_empty_raw = []
        non_empty_indices = []
        for i, p in enumerate(processed):
            if out[i] is not None:
                continue
            if p and p.strip():
                non_empty_processed.append(p)
                non_empty_raw.append(texts[i])
                non_empty_indices.append(i)
        if non_empty_processed:
            X = self._build_X(non_empty_processed, non_empty_raw)
            probas = self.model.predict_proba(X)
            for k, idx in enumerate(non_empty_indices):
                spam_score = float(probas[k][1])
                out[idx] = {"is_spam": spam_score >= self.optimal_threshold, "spam_score": spam_score}
        for i in range(len(out)):
            if out[i] is None:
                out[i] = self._empty()
        return out
