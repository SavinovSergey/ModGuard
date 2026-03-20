"""BERT модель для классификации токсичности"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.models.base import NeuralTextModelBase

logger = logging.getLogger(__name__)

# Имена ONNX файлов (квантизированная версия имеет приоритет)
_ONNX_FILES = ("model_quantized.onnx", "model.onnx")


class BERTModel(NeuralTextModelBase):
    """
    Модель классификации токсичности на основе дообученного BERT.
    Поддерживает PyTorch и квантизованную ONNX версию (автоопределение по наличию .onnx файлов).
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = 32,
        use_onnx: Optional[bool] = None,
    ):
        super().__init__(model_name="bert", model_type="bert")
        self.model_path = model_path
        self.hf_model_name = model_name  # Сохраняем имя модели из HuggingFace отдельно
        self.max_length = max_length or 512
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_onnx = use_onnx  # None = автоопределение, True = принудительно ONNX, False = PyTorch
        
        self.model: Optional[Union[AutoModelForSequenceClassification, Any]] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        self.model_params: Dict[str, Any] = {}
        self.optimal_threshold: float = 0.5
        self.batch_size: int = batch_size
        self._is_onnx: bool = False
        # Число выходов классификатора (1 = один logit, sigmoid; 2 = два logita, берём класс 1)
        self._num_labels: int = 1
    
    def _should_use_onnx(self, model_dir: Path) -> Optional[str]:
        """Проверяет наличие ONNX модели. Возвращает имя файла или None."""
        if self.use_onnx is False:
            return None
        for name in _ONNX_FILES:
            if (model_dir / name).exists():
                return name
        if self.use_onnx is True:
            raise FileNotFoundError(
                f"ONNX модель не найдена в {model_dir}. "
                f"Ожидаются файлы: {', '.join(_ONNX_FILES)}"
            )
        return None

    def _get_onnx_provider(self) -> List[str]:
        """Возвращает провайдеры ONNX Runtime для текущего device."""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
        except ImportError:
            raise ImportError(
                "onnxruntime не установлен. Установите: pip install onnxruntime\n"
                "Или для GPU: pip install onnxruntime-gpu"
            )
        
        if self.device == "cuda" or (isinstance(self.device, str) and "cuda" in self.device.lower()):
            # Пробуем CUDA, затем CPU как fallback
            providers = []
            if "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available_providers:
                providers.append("CPUExecutionProvider")
            return providers if providers else ["CPUExecutionProvider"]
        
        # Для CPU используем только CPUExecutionProvider
        if "CPUExecutionProvider" not in available_providers:
            raise RuntimeError(
                f"CPUExecutionProvider недоступен. Доступные провайдеры: {available_providers}"
            )
        return ["CPUExecutionProvider"]

    def _logits_to_score(self, logits) -> float:
        """Преобразует logits в вероятность токсичности. Поддерживает torch и numpy."""
        n = getattr(self, "_num_labels", 1)
        if hasattr(logits, "cpu"):
            arr = logits.cpu().numpy().ravel()
            idx = 1 if n == 2 and len(arr) >= 2 else 0
            return float(1.0 / (1.0 + np.exp(-arr[idx])))
        arr = np.asarray(logits).ravel()
        idx = 1 if n == 2 and len(arr) >= 2 else 0
        return float(1.0 / (1.0 + np.exp(-arr[idx])))

    def _logits_to_scores_batch(self, logits) -> np.ndarray:
        """Преобразует батч logits в вероятности (токсичность). Поддерживает torch и numpy."""
        n = getattr(self, "_num_labels", 1)
        col = 1 if n == 2 else 0  # при num_labels=1 один logit (индекс 0), при 2 — класс 1 (индекс 1)
        if hasattr(logits, "cpu"):
            logits_np = logits.cpu().numpy()
            arr = np.asarray(logits_np).squeeze()
            if arr.ndim == 1:
                raw = np.asarray(arr)
            else:
                raw = arr[:, col] if arr.ndim == 2 and arr.shape[-1] > col else arr.ravel()
            return (1.0 / (1.0 + np.exp(-raw))).astype(np.float64)
        arr = np.asarray(logits).squeeze()
        if arr.ndim == 1:
            raw = arr
        else:
            raw = arr[:, col] if arr.ndim == 2 and arr.shape[-1] > col else arr.ravel()
        return (1.0 / (1.0 + np.exp(-np.asarray(raw)))).astype(np.float64)

    def load(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None
    ) -> None:
        """
        Загружает BERT модель и токенизатор из файлов.
        Поддерживает PyTorch и квантизованную ONNX версию (автоопределение).
        
        Args:
            model_path: Путь к директории с моделью (или None, если используется model_name)
            tokenizer_path: Путь к токенизатору (обычно та же директория, что и модель)
        """
        model_path = model_path or self.model_path
        
        if model_path is None and self.hf_model_name is None:
            raise ValueError(
                "Необходимо указать либо model_path, либо model_name. "
                "Используйте load(model_path='...') или передайте model_name в __init__"
            )
        
        # Если указан model_path, загружаем из него
        if model_path is not None:
            model_dir = Path(model_path)
            if not model_dir.exists():
                raise FileNotFoundError(f"Директория модели не найдена: {model_path}")
            
            # Загружаем метаданные модели
            params_path = model_dir / 'params.json'
            if params_path.exists():
                with open(params_path, 'r', encoding='utf-8') as f:
                    self.model_params = json.load(f)
                    self.max_length = self.model_params.get('max_length', self.max_length)
                    if 'optimal_threshold' in self.model_params:
                        self.optimal_threshold = float(self.model_params['optimal_threshold'])
                        logger.info(f"Загружен optimal_threshold: {self.optimal_threshold}")
            else:
                logger.warning(f"Файл параметров не найден: {params_path}. Используются параметры по умолчанию.")
            
            onnx_file = self._should_use_onnx(model_dir)
            if onnx_file and (self.use_onnx is not False):
                # Загружаем квантизованную ONNX модель
                try:
                    from optimum.onnxruntime import ORTModelForSequenceClassification
                except ImportError as err:
                    raise ImportError(
                        "Для загрузки ONNX модели установите: pip install optimum[onnxruntime]"
                    ) from err
                
                logger.info(f"Загрузка ONNX модели из {model_path} ({onnx_file})...")
                
                # Сначала пробуем загрузить без указания провайдера (optimum использует CPU по умолчанию)
                try:
                    self.model = ORTModelForSequenceClassification.from_pretrained(
                        str(model_dir),
                        file_name=onnx_file,
                    )
                except (ValueError, RuntimeError) as e:
                    error_msg = str(e)
                    # Если ошибка связана с провайдером, пробуем указать провайдеры явно
                    if "provider" in error_msg.lower() or "execution" in error_msg.lower():
                        logger.warning(f"Ошибка при загрузке без провайдера: {error_msg}")
                        logger.info("Пробуем загрузить с явным указанием провайдеров...")
                        try:
                            providers = self._get_onnx_provider()
                            self.model = ORTModelForSequenceClassification.from_pretrained(
                                str(model_dir),
                                file_name=onnx_file,
                                provider=providers,
                            )
                        except Exception as e2:
                            raise RuntimeError(
                                f"Не удалось загрузить ONNX модель. "
                                f"Ошибка без провайдера: {error_msg}. "
                                f"Ошибка с провайдерами {providers}: {e2}. "
                                f"Убедитесь, что установлен onnxruntime: pip install onnxruntime"
                            ) from e2
                    else:
                        raise
                
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or str(model_dir))
                self._is_onnx = True  # фиксируется при загрузке, не меняется при predict/predict_batch
                self._num_labels = getattr(self.model.config, "num_labels", 1)
            else:
                # Загружаем PyTorch модель (голова — один Linear, classifier.weight/bias)
                logger.info(f"Загрузка PyTorch модели из {model_path}...")
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
                self._is_onnx = False
                self.model = self.model.to(self.device)
                self.model.eval()
                self._num_labels = getattr(self.model.config, "num_labels", 1)
        
        elif self.hf_model_name is not None:
            # Загружаем из HuggingFace (только PyTorch)
            logger.info(f"Загрузка модели {self.hf_model_name} из HuggingFace...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.hf_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            # При наличии params.json на HF-репозитории обновляем optimal_threshold/max_length.
            # Это важно для предсказаний (is_toxic) и кэширования TTL по точному порогу.
            try:
                from huggingface_hub import hf_hub_download

                params_path = hf_hub_download(
                    repo_id=self.hf_model_name,
                    filename="params.json",
                )
                if params_path:
                    with open(params_path, "r", encoding="utf-8") as f:
                        self.model_params = json.load(f)
                    self.max_length = self.model_params.get("max_length", self.max_length)
                    if "optimal_threshold" in self.model_params:
                        self.optimal_threshold = float(self.model_params["optimal_threshold"])
                        logger.info("Загружен optimal_threshold: %s", self.optimal_threshold)
            except Exception:
                # params.json может отсутствовать или быть недоступным — в этом случае остаёмся на дефолтах
                pass
            self._is_onnx = False  # фиксируется при загрузке, не меняется при predict/predict_batch
            self.model = self.model.to(self.device)
            self.model.eval()
            self._num_labels = getattr(self.model.config, "num_labels", 1)
        
        self.is_loaded = True
        logger.info(f"Модель загружена ({'ONNX' if self._is_onnx else 'PyTorch'}) на устройство: {self.device}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Предсказывает токсичность для одного текста.
        
        Args:
            text: Текст для классификации
        
        Returns:
            {
                'is_toxic': bool,
                'toxicity_score': float,  # 0-1
                'toxicity_types': Dict[str, float]  # пустой словарь для BERT
            }
        """
        self.ensure_loaded()
        
        if not text or not isinstance(text, str) or not text.strip():
            return {
                'is_toxic': False,
                'toxicity_score': 0.0,
                'toxicity_types': {}
            }
        
        # Предобработка
        processed_text = self.preprocess_text(text)
        if not processed_text.strip():
            return self.empty_result()
        
        # Токенизация
        encoding = self.tokenizer(
            processed_text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        if self._is_onnx:
            # Для ONNX модели передаем данные напрямую из токенизатора (numpy массивы)
            # ORTModelForSequenceClassification ожидает numpy массивы или torch тензоры
            outputs = self.model(
                input_ids=encoding['input_ids'].cpu().numpy(),
                attention_mask=encoding['attention_mask'].cpu().numpy(),
            )
            # ONNX модель может возвращать logits напрямую или в виде объекта с атрибутом logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                # Если outputs - это numpy массив напрямую
                logits = outputs
            toxicity_score = self._logits_to_score(logits)
        else:
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            toxicity_score = self._logits_to_score(logits)
        
        # Бинарная классификация с использованием optimal_threshold
        is_toxic = toxicity_score >= self.optimal_threshold
        
        return {
            'is_toxic': is_toxic,
            'toxicity_score': float(toxicity_score),
            'toxicity_types': {}
        }
    

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Предсказывает токсичность для батча текстов.
        Большой батч обрабатывается мини-батчами по self.batch_size (по умолчанию 32) для экономии памяти.
        """
        self.ensure_loaded()

        if not texts:
            return []

        # Предобработка текстов
        processed_texts = self.preprocess_batch(texts)

        # Фильтруем пустые тексты для токенизации
        non_empty_indices = self.non_empty_indexes(processed_texts)
        if not non_empty_indices:
            return [self.empty_result()] * len(texts)

        non_empty_texts = [processed_texts[i] for i in non_empty_indices]

        # Сортируем по длине (убывание): в мини-батче меньше паддинга, быстрее инференс
        sort_idx = sorted(range(len(non_empty_texts)), key=lambda i: len(non_empty_texts[i]), reverse=True)
        non_empty_texts = [non_empty_texts[i] for i in sort_idx]
        inv_sort = [0] * len(sort_idx)
        for k, orig in enumerate(sort_idx):
            inv_sort[orig] = k

        # Обрабатываем мини-батчами (по self.batch_size, по умолчанию 32) для экономии памяти
        chunk_size = self.batch_size
        all_toxicity_scores = []
        for start in range(0, len(non_empty_texts), chunk_size):
            chunk_texts = non_empty_texts[start : start + chunk_size]
            encoding = self.tokenizer(
                chunk_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            if self._is_onnx:
                outputs = self.model(
                    input_ids=encoding['input_ids'].cpu().numpy(),
                    attention_mask=encoding['attention_mask'].cpu().numpy(),
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                chunk_scores = self._logits_to_scores_batch(logits)
            else:
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                chunk_scores = self._logits_to_scores_batch(logits)
            if hasattr(chunk_scores, 'tolist'):
                all_toxicity_scores.extend(chunk_scores.tolist())
            else:
                all_toxicity_scores.extend(list(chunk_scores))
        # Возвращаем результаты в исходный порядок (как до сортировки по длине)
        toxicity_scores = [all_toxicity_scores[inv_sort[i]] for i in range(len(inv_sort))]

        assert len(toxicity_scores) == len(non_empty_indices), \
            f"toxicity_scores ({len(toxicity_scores)}), валидные тексты ({len(non_empty_indices)})"
        # Формируем окончательные результаты
        full_results = []
        proba_idx = 0
        for i, text in enumerate(texts):
            if i in non_empty_indices:
                toxicity_score = float(toxicity_scores[proba_idx])
                # Используем optimal_threshold вместо дефолтного порога
                is_toxic = toxicity_score >= self.optimal_threshold
                full_results.append({
                    'is_toxic': is_toxic,
                    'toxicity_score': toxicity_score,
                    'toxicity_types': {}
                })
                proba_idx += 1
            else:
                full_results.append(self.empty_result())
        return full_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели.
        
        Returns:
            {
                'name': str,
                'type': str,
                'is_loaded': bool,
                'version': str,
                'description': str
            }
        """
        info = self._base_info(
            description="BERT модель для классификации токсичности комментариев",
            version="1.0",
        )
        info.update(
            {
                'name': self.hf_model_name or (str(self.model_path) if self.model_path else self.model_name),
                'optimal_threshold': self.optimal_threshold,
                'max_length': self.max_length,
                'device': self.device,
                'is_onnx': self._is_onnx,
            }
        )
        return info
