"""Главный файл FastAPI приложения"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.model_manager import ModelManager
from app.services.classification import ClassificationService
from app.models.regex_model import RegexModel
from app.api.routes import router

from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальные переменные для зависимостей
model_manager: ModelManager = None
classification_service: ClassificationService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global model_manager, classification_service
    
    # Startup
    logger.info("Starting ToxicFilter service...")
    
    # Инициализация менеджера моделей
    model_manager = ModelManager()
    
    # Регистрация моделей
    logger.info("Registering models...")
    
    # Regex модель (всегда доступна)
    regex_model = RegexModel()
    model_manager.register_model("regex", regex_model)
    
    # TF-IDF модель (если обучена)
    try:
        from app.models.tfidf_model import TfidfModel
        
        tfidf_model = TfidfModel()
        model_path = Path("models/tfidf/model.pkl")
        vectorizer_path = Path("models/tfidf/vectorizer.pkl")
        
        if model_path.exists() and vectorizer_path.exists():
            tfidf_model.load(
                model_path=str(model_path),
                vectorizer_path=str(vectorizer_path)
            )
            model_manager.register_model("tfidf", tfidf_model)
            logger.info(f"TF-IDF model registered and loaded {tfidf_model.is_loaded}")
        else:
            logger.info("TF-IDF model files not found, skipping registration")
    except Exception as e:
        logger.warning(f"Could not register TF-IDF model: {e}")
    
    # FastText модель (если обучена)
    try:
        from app.models.fasttext_model import FastTextModel
        
        fasttext_model = FastTextModel()
        model_path = Path("models/fasttext/fasttext_model.bin")
        
        if model_path.exists():
            fasttext_model.load(model_path=str(model_path))
            model_manager.register_model("fasttext", fasttext_model)
            logger.info("FastText model registered and loaded")
        else:
            logger.info("FastText model files not found, skipping registration")
    except Exception as e:
        logger.warning(f"Could not register FastText model: {e}")
    
    # RNN модель (если обучена); поддержка квантизованной версии (model_quantized.pt)
    try:
        from app.models.rnn_model import RNNModel
        
        rnn_model = RNNModel()
        rnn_dir = Path("models/rnn")
        tokenizer_path = rnn_dir / "tokenizer.json"
        model_path = rnn_dir / "model_quantized.pt"
        if not model_path.exists():
            model_path = rnn_dir / "model.pt"
        
        if model_path.exists() and tokenizer_path.exists():
            rnn_model.load(
                model_path=str(model_path),
                tokenizer_path=str(tokenizer_path)
            )
            model_manager.register_model("rnn", rnn_model)
            logger.info(f"RNN model registered and loaded ({model_path.name})")
        else:
            logger.info("RNN model files not found, skipping registration")
    except Exception as e:
        logger.warning(f"Could not register RNN model: {e}")
    
    # BERT модель (если обучена); поддержка PyTorch и квантизованной ONNX (model_quantized.onnx / model.onnx)
    try:
        from app.models.bert_model import BERTModel
        
        bert_model = BERTModel()
        # Пробуем директории: models/bert (может содержать ONNX), models/bert_onnx, models/bert_onnx_cpu
        bert_loaded = False
        for bert_dir in (Path("models/bert"), Path("models/bert_onnx"), Path("models/bert_onnx_cpu")):
            if not bert_dir.is_dir():
                continue
            # BERTModel.load() автоопределяет ONNX по наличию model_quantized.onnx / model.onnx
            try:
                bert_model.load(model_path=str(bert_dir))
                model_manager.register_model("bert", bert_model)
                logger.info(f"BERT model registered and loaded from {bert_dir}")
                bert_loaded = True
                break
            except Exception as e_inner:
                logger.debug(f"BERT load from {bert_dir} failed: {e_inner}")
                continue
        if not bert_loaded:
            logger.info("BERT model directory not found or load failed, skipping registration")
    except ImportError as e:
        logger.warning(f"Could not import BERT model: {e}")
    except Exception as e:
        logger.warning(f"Could not register BERT model: {e}")
    
    # Загрузка модели по умолчанию (rubert — алиас для bert)
    try:
        model_manager.set_current_model(settings.model_type)
        logger.info(f"Loaded default model: {settings.model_type}")
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        # Пытаемся загрузить regex как fallback
        try:
            model_manager.set_current_model("regex")
            logger.info("Loaded regex model as fallback")
        except Exception as e2:
            logger.error(f"Failed to load fallback model: {e2}")
    
    # Инициализация сервиса классификации
    classification_service = ClassificationService(model_manager)
    
    logger.info("Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ToxicFilter service...")


# Создание FastAPI приложения
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(router, prefix=settings.api_prefix)


@app.get("/", tags=["root"])
async def root():
    """Корневой endpoint"""
    return {
        "service": "ToxicFilter",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )



