"""Загрузка моделей (общая для API и воркера)."""
import logging
from pathlib import Path
from typing import Optional

from app.core.model_manager import ModelManager
from app.core.config import settings

logger = logging.getLogger(__name__)


def get_spam_regex_model() -> "SpamRegexModel":
    """Создаёт и возвращает regex-модель для быстрого обнаружения спама."""
    from app.models.spam.regex_model import SpamRegexModel

    model = SpamRegexModel()
    model.load()
    logger.info("Spam regex pre-filter loaded (%d categories)", len(model.categories))
    return model


def get_spam_model() -> Optional["SpamTfidfModel"]:
    """Загружает TF-IDF модель спама из models/spam/ при наличии файлов. Иначе None."""
    try:
        from app.models.spam.tfidf_model import SpamTfidfModel
        model_path = Path("models/spam/tfidf/model.pkl")
        vectorizer_path = Path("models/spam/tfidf/vectorizer.pkl")
        if model_path.exists() and vectorizer_path.exists():
            spam_model = SpamTfidfModel()
            spam_model.load(
                model_path=str(model_path),
                vectorizer_path=str(vectorizer_path),
            )
            logger.info("Spam TF-IDF model loaded from models/spam/tfidf/")
            return spam_model
    except Exception as e:
        logger.warning("Could not load spam model: %s", e)
    return None


def register_all_models(model_manager: ModelManager) -> None:
    """Регистрирует все доступные модели по путям (tfidf, fasttext, rnn, bert) и regex."""
    from app.models.toxicity.regex_model import RegexModel

    regex_model = RegexModel()
    model_manager.register_model("regex", regex_model)

    try:
        from app.models.toxicity.tfidf_model import TfidfModel
        model_path = Path("models/toxicity/tfidf/model.pkl")
        vectorizer_path = Path("models/toxicity/tfidf/vectorizer.pkl")
        if model_path.exists() and vectorizer_path.exists():
            tfidf_model = TfidfModel()
            tfidf_model.load(model_path=str(model_path), vectorizer_path=str(vectorizer_path))
            model_manager.register_model("tfidf", tfidf_model)
            logger.info("TF-IDF model registered and loaded")
    except Exception as e:
        logger.warning("Could not register TF-IDF model: %s", e)

    try:
        from app.models.toxicity.fasttext_model import FastTextModel
        model_path = Path("models/toxicity/fasttext/fasttext_model.bin")
        if model_path.exists():
            fasttext_model = FastTextModel()
            fasttext_model.load(model_path=str(model_path))
            model_manager.register_model("fasttext", fasttext_model)
            logger.info("FastText model registered and loaded")
    except Exception as e:
        logger.warning("Could not register FastText model: %s", e)

    try:
        from app.models.toxicity.rnn_model import RNNModel
        rnn_dir = Path("models/toxicity/rnn")
        tokenizer_path = rnn_dir / "tokenizer.json"
        model_path = rnn_dir / "model_quantized.pt"
        if not model_path.exists():
            model_path = rnn_dir / "model.pt"
        if model_path.exists() and tokenizer_path.exists():
            rnn_model = RNNModel()
            rnn_model.load(model_path=str(model_path), tokenizer_path=str(tokenizer_path))
            model_manager.register_model("rnn", rnn_model)
            logger.info("RNN model registered and loaded")
    except Exception as e:
        logger.warning("Could not register RNN model: %s", e)

    try:
        from app.models.toxicity.bert_model import BERTModel
        bert_model = BERTModel()
        bert_loaded = False
        for bert_dir in (Path("models/toxicity/bert"), Path("models/toxicity/bert/onnx"), Path("models/toxicity/bert/onnx_cpu")):
            if not bert_dir.is_dir():
                continue
            try:
                bert_model.load(model_path=str(bert_dir))
                model_manager.register_model("bert", bert_model)
                logger.info("BERT model registered and loaded from %s", bert_dir)
                bert_loaded = True
                break
            except Exception as e_inner:
                logger.debug("BERT load from %s failed: %s", bert_dir, e_inner)
        if not bert_loaded:
            logger.info("BERT model directory not found or load failed")
    except ImportError as e:
        logger.warning("Could not import BERT model: %s", e)
    except Exception as e:
        logger.warning("Could not register BERT model: %s", e)

    try:
        model_manager.set_current_model(settings.model_type)
        logger.info("Default model set: %s", settings.model_type)
    except Exception as e:
        logger.error("Failed to load default model: %s", e)
        try:
            model_manager.set_current_model("regex")
            logger.info("Loaded regex model as fallback")
        except Exception as e2:
            logger.error("Failed to load fallback model: %s", e2)
