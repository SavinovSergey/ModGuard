"""RNN модель для классификации токсичности"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from app.models.base import NeuralTextModelBase
from app.models.toxicity.rnn_network import RNNClassifier
from app.models.toxicity.rnn_tokenizers import create_tokenizer, BaseRNNTokenizer
from app.models.toxicity.rnn_dataset import collate_fn

logger = logging.getLogger(__name__)


class RNNModel(NeuralTextModelBase):
    """
    Модель классификации токсичности на основе рекуррентных нейронных сетей (RNN, GRU, LSTM).
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        tokenizer_type: str = 'bpe',  # 'bpe' or 'rubert'
        rnn_type: str = 'gru',  # 'rnn', 'gru', 'lstm'
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = 64
    ):
        super().__init__(model_name="rnn", model_type="rnn")
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = tokenizer_type
        self.rnn_type = rnn_type
        self.max_length = max_length

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[RNNClassifier] = None
        self.tokenizer: Optional[BaseRNNTokenizer] = None

        self.model_params: Dict[str, Any] = {}  # Для хранения параметров модели из params.json
        self.optimal_threshold: float = 0.5  # Порог по умолчанию
        self.batch_size = batch_size

    def load(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None
    ) -> None:
        """
        Загружает RNN модель и токенизатор из файлов.
        """
        model_path = model_path or self.model_path
        tokenizer_path = tokenizer_path or self.tokenizer_path

        if model_path is None or tokenizer_path is None:
            raise ValueError(
                "Необходимо указать пути к модели и токенизатору. "
                "Используйте load(model_path='...', tokenizer_path='...')"
            )

        # Проверяем существование файлов
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        # Определяем директорию модели (может быть файл или директория)
        if model_path_obj.is_file():
            model_dir = model_path_obj.parent
            # Если это файл model_quantized.pt, используем его напрямую
            is_quantized_file = model_path_obj.name == "model_quantized.pt"
        else:
            model_dir = model_path_obj
            is_quantized_file = False

        # Загружаем метаданные модели
        params_path = model_dir / 'params.json'
        if params_path.exists():
            with open(params_path, 'r', encoding='utf-8') as f:
                self.model_params = json.load(f)
                self.tokenizer_type = self.model_params.get('tokenizer_type', self.tokenizer_type)
                self.rnn_type = self.model_params.get('rnn_type', self.rnn_type)
                self.max_length = self.model_params.get('max_length', self.max_length)
                # Загружаем optimal_threshold если доступен
                if 'optimal_threshold' in self.model_params:
                    self.optimal_threshold = float(self.model_params['optimal_threshold'])
                    logger.info(f"Загружен optimal_threshold: {self.optimal_threshold}")
        else:
            logger.warning(f"Файл параметров не найден: {params_path}. Используются параметры по умолчанию.")
            logger.info(f"Используется порог по умолчанию: {self.optimal_threshold}")

        # Загружаем токенизатор
        self.tokenizer = create_tokenizer(self.tokenizer_type)
        self.tokenizer.load(tokenizer_path)

        # Создаем модель с правильными параметрами
        vocab_size = self.tokenizer.get_vocab_size()
        embedding_dim = self.model_params.get('embedding_dim', 100)
        hidden_size = self.model_params.get('hidden_size', 128)
        num_layers = self.model_params.get('num_layers', 1)
        dropout = self.model_params.get('dropout', 0.2)
        bidirectional = self.model_params.get('bidirectional', False)
        embedding_dropout = self.model_params.get('embedding_dropout', 0.0)
        use_layer_norm = self.model_params.get('use_layer_norm', False)

        self.model = RNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=self.rnn_type,
            dropout=dropout,
            bidirectional=bidirectional,
            padding_idx=self.tokenizer.get_pad_token_id(),
            embedding_dropout=embedding_dropout,
            use_layer_norm=use_layer_norm
        )

        # Проверяем, есть ли квантизированная версия модели
        if is_quantized_file:
            # Если model_path указывает на model_quantized.pt, используем его напрямую
            actual_model_path = model_path_obj
            is_quantized = True
            logger.info(f"Загрузка квантизированной модели из {actual_model_path}...")
        else:
            # Иначе проверяем наличие квантизированной версии в той же директории
            quantized_model_path = model_dir / "model_quantized.pt"
            is_quantized = self.model_params.get("quantized", False) and quantized_model_path.exists()
            
            if is_quantized:
                logger.info(f"Загрузка квантизированной модели из {quantized_model_path}...")
                actual_model_path = quantized_model_path
            else:
                # Определяем путь к оригинальной модели
                if model_path_obj.is_file():
                    actual_model_path = model_path_obj
                else:
                    actual_model_path = model_dir / "model.pt"
        
        # Загружаем веса
        # Для квантизированных моделей может потребоваться weights_only=False
        # так как они могут содержать дополнительные метаданные
        try:
            state_dict = torch.load(actual_model_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Для старых версий PyTorch параметр weights_only не поддерживается
            state_dict = torch.load(actual_model_path, map_location=self.device)
        
        if is_quantized:
            # Для квантизированных моделей нужно сначала применить квантизацию,
            # а затем загрузить state_dict. Квантизированные слои имеют другую структуру.
            try:
                # Пробуем импортировать функцию квантизации
                try:
                    from torch.ao.quantization import quantize_dynamic
                except ImportError:
                    from torch.quantization import quantize_dynamic
                
                from torch.nn import Linear, LSTM, GRU, RNN
                
                # Применяем квантизацию к модели перед загрузкой весов
                quantization_dtype_str = self.model_params.get("quantization_dtype", "torch.qint8")
                if "qint8" in quantization_dtype_str:
                    quantization_dtype = torch.qint8
                elif "float16" in quantization_dtype_str or "fp16" in quantization_dtype_str:
                    quantization_dtype = torch.float16
                else:
                    quantization_dtype = torch.qint8
                
                logger.info(f"Применение квантизации (dtype={quantization_dtype}) перед загрузкой весов...")
                self.model = quantize_dynamic(
                    self.model,
                    {Linear, LSTM, GRU, RNN},
                    dtype=quantization_dtype,
                )
                
                # Применяем квантизацию эмбеддингов в FP16, если она была применена
                if self.model_params.get("embeddings_quantized_fp16", False):
                    if hasattr(self.model, 'embedding') and hasattr(self.model.embedding, 'weight'):
                        with torch.no_grad():
                            self.model.embedding.weight.data = self.model.embedding.weight.data.half()
                        logger.info("Эмбеддинги квантизированы в FP16")
                
            except Exception as e:
                logger.warning(f"Не удалось применить квантизацию перед загрузкой: {e}. Пробуем загрузить напрямую.")
        
        # Загружаем state_dict (для квантизированных моделей может потребоваться strict=False)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            if is_quantized:
                # Для квантизированных моделей пробуем загрузить с strict=False
                logger.warning(f"Не удалось загрузить state_dict строго: {e}. Пробуем с strict=False.")
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logger.warning(f"Отсутствующие ключи при загрузке квантизированной модели: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Неожиданные ключи при загрузке квантизированной модели: {unexpected_keys}")
            else:
                raise
        
        self.model.to(self.device)
        self.model.eval()  # Переводим модель в режим оценки

        # Сохраняем пути
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.is_loaded = True
        logger.info(f"RNN модель загружена из {model_path}, токенизатор из {tokenizer_path}, устройство: {self.device}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Предсказывает токсичность для одного текста
        """
        self.ensure_loaded()

        # Предобработка текста (только нормализация, без лемматизации)
        processed_text = self.preprocess_text(text)

        if not processed_text or not processed_text.strip():
            return self.empty_result()

        # Токенизация
        token_ids = self.tokenizer.encode(processed_text, max_length=self.max_length)

        # Преобразуем в тензор
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        pad_token_id = self.tokenizer.get_pad_token_id()
        length = int((input_tensor[0] != pad_token_id).sum().item())
        lengths_tensor = torch.tensor([max(1, length)], dtype=torch.long)

        # Предсказание
        with torch.no_grad():
            logits = self.model(input_tensor, lengths=lengths_tensor)
            proba = torch.sigmoid(logits).cpu().item()

        toxicity_score = float(proba)
        # Используем optimal_threshold вместо дефолтного порога
        is_toxic = toxicity_score >= self.optimal_threshold

        return {
            'is_toxic': is_toxic,
            'toxicity_score': toxicity_score,
            'toxicity_types': {}
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Предсказывает токсичность для батча текстов
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

        # Токенизация батча
        token_ids_batch = self.tokenizer.encode_batch(non_empty_texts, max_length=self.max_length)

        # Разбиваем на батчи и обрабатываем напрямую через collate_fn
        pad_token_id = self.tokenizer.get_pad_token_id()
        
        all_probas = []
        self.model.eval()
        with torch.no_grad():
            # Обрабатываем батчами
            for i in range(0, len(token_ids_batch), self.batch_size):
                batch_token_ids = token_ids_batch[i:i + self.batch_size]
                
                # Создаем список словарей для collate_fn
                batch_items = [
                    {
                        'input_ids': torch.tensor(token_ids, dtype=torch.long),
                        'label': torch.tensor(0.0, dtype=torch.float)  # Фиктивная метка, не используется
                    }
                    for token_ids in batch_token_ids
                ]
                
                # Применяем collate_fn напрямую
                batch_data = collate_fn(batch_items, pad_token_id=pad_token_id)
                
                # Переносим на устройство и делаем предсказание
                input_ids = batch_data['input_ids'].to(self.device)
                lengths = batch_data['lengths'].cpu()
                logits = self.model(input_ids, lengths=lengths)
                probas = torch.sigmoid(logits).cpu().squeeze().tolist()
                if isinstance(probas, float):
                    probas = [probas]
                all_probas.extend(probas)
        
        # Формируем окончательные результаты
        full_results = []
        proba_idx = 0
        for i, text in enumerate(texts):
            if i in non_empty_indices:
                proba = all_probas[proba_idx]
                toxicity_score = float(proba)
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
        """Возвращает информацию о модели"""
        info = self._base_info("RNN-based toxicity classification model (RNN/GRU/LSTM)")
        info.update(
            {
                'tokenizer_type': self.tokenizer_type,
                'rnn_type': self.rnn_type,
                'max_length': self.max_length,
                'device': str(self.device),
            }
        )
        
        # Добавляем информацию о квантизации и параметры модели
        if self.is_loaded:
            if self.model_params:
                # Используем параметры из params.json
                info['model_params'] = self.model_params
                # Добавляем информацию о квантизации
                info['is_quantized'] = self.model_params.get('quantized', False)
                if info['is_quantized']:
                    info['quantization_dtype'] = self.model_params.get('quantization_dtype', 'unknown')
                    info['quantization_method'] = self.model_params.get('quantization_method', 'unknown')
                    info['embeddings_quantized_fp16'] = self.model_params.get('embeddings_quantized_fp16', False)
                if self.tokenizer:
                    info['vocab_size'] = self.tokenizer.get_vocab_size()
            elif self.model:
                # Если params.json не был загружен, пытаемся получить из модели
                info['model_params'] = {
                    'embedding_dim': self.model.embedding_dim,
                    'hidden_size': self.model.hidden_size,
                    'num_layers': self.model.num_layers,
                    'dropout': self.model.dropout,
                    'bidirectional': self.model.bidirectional,
                    'vocab_size': self.tokenizer.get_vocab_size() if self.tokenizer else None
                }
                info['is_quantized'] = False
            else:
                # Если модель не загружена, но is_loaded=True (не должно происходить, но на всякий случай)
                info['model_params'] = {}
                info['is_quantized'] = False
        return info
