# ModGuard — система модерации контента

ModGuard — система модерации контента для автоматической классификации **токсичности** и **спама** в комментариях
и сообщениях. Поддержка различных моделей машинного обучения, единый API и асинхронная обработка через очереди.

## Архитектура

Проект использует модульную архитектуру с поддержкой:
- Модульной замены моделей (токсичность и спам)
- Автоматического fallback при ошибках/таймаутах
- Real-time и batch обработки
- Единого ответа модерации (оценки токсичности и спама)

Подробное описание целевой архитектуры (очереди, Backend, Listeners, Action-сервисы, кэш): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Структура проекта

Краткое дерево и назначение каталогов: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

```
├── app/           # API, core, модели (toxicity, spam), preprocessing, services, frontend
├── scripts/       # Обучение (toxicity, spam), запуск воркера/listener/actions, healthcheck
├── tests/         # Тесты API, моделей токсичности и спама
├── docs/          # Документация
├── docker/        # Docker конфигурация
└── models/        # Артефакты обученных моделей
```

## Требования

- Python 3.11+
- Docker и Docker Compose (для контейнерного запуска)
- Redis 7+, PostgreSQL 15+, RabbitMQ 3+ (поднимаются автоматически через Docker Compose или устанавливаются вручную)

## Быстрый старт (Docker) — рекомендуемый способ

Docker Compose поднимает все сервисы (API, worker, Redis, PostgreSQL, RabbitMQ, фронтенд) одной командой.

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/SergeySavinov/ModGuard.git
cd ModGuard

# 2. Скопируйте пример конфигурации и при необходимости отредактируйте
cp .env.example .env

# 3. Соберите образы и запустите
docker compose build
docker compose up -d
```

После запуска:

| Сервис | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Демо-чат | http://localhost:8080 |
| RabbitMQ Management | http://localhost:15672 (guest / guest) |

По умолчанию worker собирается с минимальными зависимостями (TF-IDF + regex, без PyTorch).  
Для сборки со всеми моделями: `docker compose build backend --build-arg DEPS=full`.  
Для GPU: `docker compose build backend --build-arg DEPS=full --build-arg DEVICE=gpu`.  
Для подключения Telegram-бота: `docker compose --profile telegram up -d` (предварительно указав 
`TELEGRAM_BOT_TOKEN` в `.env`).

Подробнее: [docker/README.md](docker/README.md).

## Локальная установка (для разработки)

### 1. Создайте виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Установите зависимости

Зависимости разбиты на профили в папке `requirements/` — устанавливайте только то, что нужно:

| Файл | Что включает | Когда нужен |
|---|---|---|
| `requirements/core.txt` | FastAPI, sklearn, NLP-препроцессинг, Redis/RabbitMQ/Postgres | API + worker (TF-IDF/regex) |
| `requirements/data.txt` | + pandas, datasets, matplotlib, tqdm | Загрузка данных, валидация (`validate_*.py`) |
| `requirements/train.txt` | + optuna, accelerate, sentencepiece | Обучение моделей |
| `requirements/torch.txt` | + transformers, tokenizers | RNN и BERT-PyTorch модели |
| `requirements/bert.txt` | + onnxruntime, optimum | BERT ONNX inference |
| `requirements/fasttext.txt` | + fasttext | FastText-модель |
| `requirements/test.txt` | pytest, pytest-cov, pytest-asyncio | Тесты |
| `requirements/all.txt` | Все вышеперечисленные | Полная установка |

**Минимум для запуска сервиса (TF-IDF + regex):**

```bash
pip install -r requirements/core.txt
```

**Для запуска валидации (`validate_*.py`) и подготовки данных:**

```bash
pip install -r requirements/data.txt -r requirements/test.txt
```

**Для работы с PyTorch-моделями (RNN, BERT):**

```bash
# Сначала PyTorch (CPU или GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# или GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements/torch.txt
```

**Полная установка (все модели + обучение + тесты):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/all.txt
```

### 4. Поднимите инфраструктуру

Redis, PostgreSQL и RabbitMQ необходимы для полноценной работы (batch-обработка, кэш, хранение результатов).  
Проще всего поднять их через Docker Compose, не запуская приложение в контейнере:

```bash
docker compose up -d redis postgres rabbitmq
```

Или установите и запустите их локально. Убедитесь, что параметры подключения в `.env` соответствуют вашей среде:

```
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/modguard
REDIS_URL=redis://localhost:6379/0
```

### PostgreSQL: база и таблицы

Схема хранится в коде (`app/core/db.py`). При **первом** успешном запуске **API** или **воркера** вызывается `init_db()`:

1. если базы из `DATABASE_URL` ещё нет на сервере, приложение подключается к служебной БД `postgres` и выполняет 
`CREATE DATABASE` (нужны права `CREATEDB` или суперпользователь);
2. создаются таблицы `tasks` и `task_items` (`CREATE TABLE IF NOT EXISTS ...`).

Вручную из корня репозитория (после настройки `.env`):

```bash
python scripts/run/init_postgres.py
```

**Docker:** при первом создании тома контейнер Postgres создаёт БД из `POSTGRES_DB` в `docker-compose.yml`. Если том уже был 
инициализирован под другое имя БД, либо пересоздайте том, либо выполните `init_postgres.py` / создайте БД вручную.

### 5. Запустите сервисы

Необходимо запустить **API-сервер** и **worker** в отдельных терминалах:

```bash
# Терминал 1 — API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Терминал 2 — Worker (обработка очереди)
python scripts/run/run_worker.py
```

Сервис будет доступен по адресу: http://localhost:8000  
**Чат-демо:** http://localhost:8000/chat

## API

Эндпоинты возвращают результаты по **токсичности** и по **спаму** (поля `toxicity_score`, `is_toxic`, 
`spam_score`, `is_spam` и др.).

### Классификация одного сообщения
```bash
POST /api/v1/classify
{
  "text": "Текст комментария",
  "preferred_model": "regex"   # опционально, для токсичности
}
```

### Batch классификация (асинхронная)
```bash
POST /api/v1/classify/batch-async
{
  "items": [{"text": "Текст 1"}, {"text": "Текст 2"}],
  "preferred_model": "regex"   # опционально
}
# Ответ: task_id. Результат получать опросом GET /api/v1/tasks/{task_id} до статуса completed.
```

### Получение результата задачи
```bash
GET /api/v1/tasks/{task_id}
```

### Health check
```bash
GET /api/v1/health
```

### Статистика
```bash
GET /api/v1/stats
```

После запуска сервиса доступна автоматическая документация:
- **Swagger UI:** http://localhost:8000/docs  
- **ReDoc:** http://localhost:8000/redoc

## Модели

### Токсичность
- **Regex** — базовая модель на регулярных выражениях  
- **TF-IDF** — TF-IDF + классификатор  
- **FastText** — эмбеддинги FastText  
- **RNN** — LSTM/GRU  
- **rubert-tiny** — fine-tuned трансформер (в т.ч. ONNX)

### Спам
- **Regex** — правила (earnings, cta_links, recruitment)
- **TF-IDF** — `char_wb (3,4)` + ручные признаки (`length_chars_log1p`, `space_ratio`)

Артефакты обученных моделей сохраняются в каталоге `models/` (подкаталоги `models/toxicity/`, `models/spam/`). 
Примеры обучения и запуска скриптов: [docs/SCRIPTS.md](docs/SCRIPTS.md).

## Документация

- [Дизайн-документ ModGuard](docs/DESIGN.md) — истоки, проблематика, цели, требования, план развития  
- [Архитектура](docs/ARCHITECTURE.md) — компоненты, очереди, Backend, Action-сервисы, кэш, батчи  
- [Структура проекта](docs/PROJECT_STRUCTURE.md) — дерево каталогов и назначение  
- [Скрипты](docs/SCRIPTS.md) — описание и примеры запуска скриптов обучения и сервисов  
- [Бенчмарки: методика и результаты](benchmarks/README.md) — throughput, latency, команды воспроизведения

## Оценка на тесте (without threshold tuning)

Для оценки на `test`-наборе используйте режим `--eval-mode test`:

```bash
# Токсичность: используем сохранённый model.optimal_threshold (без подбора)
python scripts/toxicity/validate_toxicity.py \
  --model-type bert \
  --model-dir SergeySavinov/rubert-tiny-toxicity \
  --val-data data/test.parquet \
  --eval-mode test

# Спам: используем сохранённый model.optimal_threshold (без подбора)
python scripts/spam/validate_spam.py \
  --model-type tfidf \
  --model-dir models/spam/tfidf \
  --val-data spam_data/test.parquet \
  --eval-mode test
```

Таблица для токсичности (metrics считаются на фиксированном пороге).

| Модель токсичности | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| `regex` | 0.996 | 0.702 | 0.823 | - |
| `tfidf` | 0.941 | 0.883 | 0.911 | 0.954 |
| `fasttext` | 0.977 | 0.756 | 0.855 | 0.937 |
| `rnn` | 0.946 | 0.880 | 0.912 | 0.950 |
| `bert` | 0.948 | 0.880 | 0.913 | 0.954 |

Таблица для спама.

| Модель спама | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| `tfidf` | 0.975 | 0.968 | 0.972 | 0.994 |

## Производительность

Замеры на полном стенде **API → RabbitMQ → worker → PostgreSQL/Redis**  
(4× backend, TF-IDF tox/spam `char_wb (3,4)`, pipeline `both`, cold cache).  
Подробная методика, условия и команды: **[benchmarks/README.md](benchmarks/README.md)**.

| Метрика | Значение |
|---------|----------|
| **Пиковая throughput (steady, PG)** | **≈ 15.9k msg/s** |
| Single-request latency (server, cold) | **p50 7 ms** |
| **Batch latency (prod)** | batch=1000, **10k msg/s**: **p50 175 ms**, p99 242 ms |
| Рабочая зона (p99 batch e2e < 300 ms) | **10k msg/s** при batch=1000 (≈63% C) |

Замеры prod-сценария: open-loop, `POST /classify/batch-async`, batch=1000,
cold cache, 4× backend. Capacity-дамп (`batch_size=1000`) даёт потолок ~16k msg/s;
batch e2e при перегрузке ~2 s — метрика потолка, не SLA.

Стек I/O: asyncpg, aio-pika, redis.asyncio; worker async с ProcessPool для ML.

```bash
# Prod: batch=1000 при 10k msg/s
.venv/bin/python scripts/run/validate_chain.py \
  --val-data data/toxicity/val.parquet -n 50000 \
  --batch-size 1000 --target-rate 10000 --arrival poisson \
  --duplicate-ratio 0 --send-workers 16 --clear-cache \
  --run-tag load-10k-batch1000 \
  --backend-replicas 4 --pool-workers 1 --prefetch 12
```

## Конфигурация

Настройки задаются через переменные окружения (см. `.env.example`):

- `MODEL_TYPE` — тип модели токсичности по умолчанию (regex, tfidf, fasttext, rnn, rubert)
- `LOG_LEVEL` — уровень логирования (INFO, DEBUG, WARNING, ERROR)
- `MAX_BATCH_SIZE` — максимальный размер батча (по умолчанию 1000)
- `RABBITMQ_URL`, `DATABASE_URL`, `REDIS_URL` — для воркера, listener и actions (см. docs/ARCHITECTURE.md)

## Разработка

### Добавление новой модели токсичности

1. Создайте класс модели, наследующий `BaseToxicityModel` (см. `app/models/base.py`).
2. Зарегистрируйте модель в `app/loader.py` и добавьте в fallback chain в конфигурации.

## Тестирование

```bash
pytest tests/

# С покрытием
pytest tests/ --cov=app
```

## Лицензия

MIT
