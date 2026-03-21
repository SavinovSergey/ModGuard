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

## Быстрый старт

### Локальный запуск

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Запустите сервис:
```bash
python -m app.main
```
или:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Сервис будет доступен по адресу: http://localhost:8000  
**Чат-демо (модерация: токсичность и спам):** http://localhost:8000/chat

### Docker

1. Соберите образы:
```bash
docker compose build
```

2. Запустите контейнеры:
```bash
docker compose up -d
```
API: http://localhost:8000, демо-чат: http://localhost:8080. Подробнее: [docker/README.md](docker/README.md).

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
- **Regex** — правила (в т.ч. CAPS_WORD)  
- **TF-IDF** — TF-IDF + ручные признаки (капс, URL, email и т.д.)

Артефакты обученных моделей сохраняются в каталоге `models/` (подкаталоги `models/toxicity/`, `models/spam/`). 
Примеры обучения и запуска скриптов: [docs/SCRIPTS.md](docs/SCRIPTS.md).

## Документация

- [Дизайн-документ ModGuard](docs/DESIGN.md) — истоки, проблематика, цели, требования, план развития  
- [Архитектура](docs/ARCHITECTURE.md) — компоненты, очереди, Backend, Action-сервисы, кэш, батчи  
- [Структура проекта](docs/PROJECT_STRUCTURE.md) — дерево каталогов и назначение  
- [Скрипты](docs/SCRIPTS.md) — описание и примеры запуска скриптов обучения и сервисов  

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

| Модель токсичности | Порог | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|
| `regex` | - | 0.973 | 0.724 | 0.830 | - |
| `tfidf` | `model.optimal_threshold` | 0.911 | 0.913 | 0.912 | 0.958 |
| `fasttext` | `model.optimal_threshold` | 0.953 | 0.857 | 0.903 | 0.949 |
| `rnn` | `model.optimal_threshold` | 0.920 | 0.909 | 0.914 | 0.956 |
| `bert` | `model.optimal_threshold` | 0.954 | 0.905 | 0.928 | 0.963 |

Таблица для спама.

| Модель спама | Порог | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|
| `tfidf` | `model.optimal_threshold` | 0.975 | 0.968 | 0.972 | 0.994 |

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
