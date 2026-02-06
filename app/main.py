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
    regex_model = RegexModel()
    model_manager.register_model("regex", regex_model)
    
    # Загрузка модели по умолчанию
    try:
        model_manager.load_model(settings.model_type)
        logger.info(f"Loaded default model: {settings.model_type}")
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        # Пытаемся загрузить regex как fallback
        try:
            model_manager.load_model("regex")
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



