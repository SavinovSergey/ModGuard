# Многоступенчатый Dockerfile для ToxicFilter API
# Используем Python 3.10 для совместимости с FastText и numpy < 2.0

# Этап 1: Базовый образ с зависимостями
FROM python:3.10-slim as base

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование requirements.txt
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Этап 2: Финальный образ
FROM python:3.10-slim

# Установка только необходимых системных зависимостей для runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя для запуска приложения (безопасность)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Установка рабочей директории
WORKDIR /app

# Копирование установленных пакетов из базового образа
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Копирование кода приложения
COPY app/ ./app/
COPY run.py .

# Создание директории для моделей (будет смонтирована через volume)
RUN mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Переключение на пользователя appuser
USER appuser

# Переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_TYPE=regex \
    LOG_LEVEL=INFO \
    API_PREFIX=/api/v1 \
    HOST=0.0.0.0 \
    PORT=8000

# Открытие порта
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Команда запуска
CMD ["python", "run.py"]








