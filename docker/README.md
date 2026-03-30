# Docker конфигурация для ModGuard

Сборка и запуск через **docker-compose** в корне проекта. Отдельные Dockerfile используются для сервисов API, backend (воркер) и фронтенда.

## Файлы

| Файл | Назначение |
|------|------------|
| `docker/Dockerfile.api` | Образ для API и для listener/actions (Python, FastAPI). |
| `docker/Dockerfile.backend` | Образ воркера классификации (полный ML-стек: токсичность и спам). |
| `docker/frontend/Dockerfile` | Статика фронтенда (nginx). |
| `docker/frontend/nginx.conf` | Конфигурация nginx для фронта. |

## Запуск через docker-compose

Из **корня проекта**:

```bash
# Сборка всех образов
docker compose build

# Запуск основных сервисов (API, frontend, backend, redis, postgres, rabbitmq)
docker compose up -d

# С профилем Telegram (добавляет listener-telegram и actions-telegram)
docker compose --profile telegram up -d
```

После запуска:
- **API:** http://localhost:8000 (документация: http://localhost:8000/docs)
- **Фронтенд (демо-чат):** http://localhost:8080
- **RabbitMQ Management:** http://localhost:15672 (guest/guest)

## Сервисы в docker-compose

| Сервис | Назначение |
|--------|------------|
| **api** | FastAPI: приём запросов, проверка кэша, постановка в очередь, GET /tasks/{task_id}. |
| **backend** | Воркер классификации: очередь запросов → токсичность + спам → Postgres/Redis, очередь результатов. |
| **frontend** | Демо веб-интерфейс (чат). |
| **listener-telegram** | Приём событий Telegram, публикация в очередь (профиль `telegram`). |
| **actions-telegram** | Действия по результатам модерации в Telegram (профиль `telegram`). |
| **redis** | Кэш модерации (с AOF). |
| **postgres** | БД для задач и результатов. |
| **rabbitmq** | Очереди запросов и результатов. |

## Переменные окружения

В docker-compose для сервисов заданы:
- `REDIS_URL=redis://redis:6379/0`
- `DATABASE_URL=postgresql://postgres:postgres@postgres:5432/modguard`
- `RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/`

При старте **API** и **воркера** вызывается `init_db()` (таблицы `tasks` / `task_items`; при необходимости создаётся БД из имени в `DATABASE_URL`). Вручную: `python scripts/run/init_postgres.py` в корне репозитория.

Для API и воркера также: `MODEL_TYPE`, `LOG_LEVEL`, `API_PREFIX`. Для Telegram-сервисов: `TELEGRAM_BOT_TOKEN` (из `.env` или окружения).

## Volumes

- `./models` → `/app/models` в контейнере backend (артефакты моделей).
- По умолчанию данные Redis и Postgres хранятся в именованных томах `redis_data`, `postgres_data`.

## Сборка воркера с GPU

```bash
docker compose build backend --build-arg DEVICE=gpu
```

Используется тот же `docker/Dockerfile.backend` с аргументом `DEVICE=gpu` (устанавливается PyTorch с CUDA).

## Healthcheck

- **api:** GET http://localhost:8000/api/v1/health
- **backend:** скрипт `scripts/run/healthcheck_worker.py`
- **listener-telegram / actions-telegram:** проверка RabbitMQ

Подробнее об архитектуре и скриптах: [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md), [docs/SCRIPTS.md](../docs/SCRIPTS.md).
