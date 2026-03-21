# Скрипты ModGuard

Краткое описание скриптов и примеры запуска. Все команды выполняются из **корня проекта** (каталог с `app/`, `scripts/`, `models/`).

---

## Обучение моделей токсичности

| Скрипт | Назначение |
|--------|------------|
| `scripts/toxicity/train_bert.py` | Дообучение BERT (ruBERT-tiny2 и др.) для классификации токсичности; опционально экспорт в ONNX. |
| `scripts/toxicity/train_tfidf.py` | Обучение TF-IDF + LogisticRegression с Optuna. |
| `scripts/toxicity/train_rnn.py` | Обучение RNN (LSTM/GRU) для токсичности. |
| `scripts/toxicity/train_fasttext.py` | Обучение FastText модели. |

### Примеры запуска (токсичность)

```bash
# BERT: обучение с train/val parquet
python scripts/toxicity/train_bert.py \
  --train-data data/train.parquet \
  --val-data data/val.parquet \
  --output-dir models/toxicity/bert \
  --epochs 3 --batch-size 16

# BERT с последующей квантизацией в ONNX
python scripts/toxicity/train_bert.py \
  --train-data data/train.parquet \
  --val-data data/val.parquet \
  --output-dir models/toxicity/bert \
  --quantize-onnx --quantize-device cpu

# TF-IDF (общие аргументы: --data для CV или --train-data/--val-data)
python scripts/toxicity/train_tfidf.py \
  --train-data data/train.parquet \
  --val-data data/val.parquet \
  --output-dir models/toxicity/tfidf
```

---

## Квантизация и экспорт (токсичность)

| Скрипт | Назначение |
|--------|------------|
| `scripts/toxicity/quantize_bert_onnx.py` | Экспорт BERT в ONNX и опциональная квантизация (CPU/GPU). |
| `scripts/toxicity/quantize_rnn.py` | Динамическая квантизация RNN модели (int8). |

### Примеры

```bash
# BERT: экспорт в ONNX с квантизацией для CPU (AVX2)
python scripts/toxicity/quantize_bert_onnx.py models/toxicity/bert \
  -o models/toxicity/bert/onnx_cpu \
  --device cpu

# BERT: экспорт из HuggingFace model-id (без локальной папки)
python scripts/toxicity/quantize_bert_onnx.py SergeySavinov/rubert-tiny-toxicity \
  --export-only \
  -o models/toxicity/bert/onnx_from_hf

# BERT: только экспорт без квантизации
python scripts/toxicity/quantize_bert_onnx.py models/toxicity/bert \
  --export-only \
  -o models/toxicity/bert/onnx

# RNN: квантизация
python scripts/toxicity/quantize_rnn.py models/toxicity/rnn/model.pt models/toxicity/rnn/tokenizer.json \
  -o models/toxicity/rnn

# RNN: загрузка артефактов из HuggingFace model-id (без локальных файлов)
python scripts/toxicity/quantize_rnn.py SergeySavinov/rurnn-toxicity -o models/toxicity/rnn_from_hf
```

---

## Валидация моделей токсичности

| Скрипт | Назначение |
|--------|------------|
| `scripts/toxicity/validate_toxicity.py` | Валидация модели на val-данных: PR-кривая, подбор порога (max F1 при precision ≥ 0.90), анализ FP/FN. |

### Пример

```bash
python scripts/toxicity/validate_toxicity.py \
  --model-type bert \
  --model-dir models/toxicity/bert \
  --val-data data/val.parquet \
  --output-dir models/toxicity/bert
```

Поддерживаемые `--model-type`: `bert`, `tfidf`, `rnn`, `fasttext`, `regex` (и др. в зависимости от кода).

---

## Валидация моделей спама

| Скрипт | Назначение |
|--------|------------|
| `scripts/spam/validate_spam.py` | Валидация спам-модели на val-данных: PR-кривая, подбор порога (max F1 при precision ≥ 0.90), анализ FP/FN, при необходимости — важность признаков и обновление `params.json`. |
| `scripts/spam/eval_spam_rules.py` | Оценка правил спама (например CAPS_WORD) на данных: Precision/Recall по правилам. |

Режимы `validate_spam.py` (флаг `--model-type`):
- **tfidf** — полный пайплайн (TF-IDF + ручные признаки + правило caps+!!): 
PR-кривая, порог, FP/FN с признаками, опционально обновление порога в `params.json`.
- **regex** — только regex pre-filter: Precision/Recall/F1 при фиксированном пороге, 
FP/FN с указанием сработавших категорий (earnings, cta_links, casino и т.д.).

### Примеры (валидация спама)

```bash
# Валидация TF-IDF спам-модели (требуется matplotlib для графиков)
python scripts/spam/validate_spam.py --model-type tfidf --val-data spam_data/val.parquet

# Валидация с указанием директории модели и сохранением ошибок
python scripts/spam/validate_spam.py --model-type tfidf --val-data spam_data/val.parquet \
  --model-dir models/spam/tfidf --errors-output models/spam/tfidf/val_errors.csv

# Валидация regex-правил спама
python scripts/spam/validate_spam.py --model-type regex --val-data spam_data/val.parquet
```

---

## Обучение моделей спама

| Скрипт | Назначение |
|--------|------------|
| `scripts/spam/train_spam.py` | Обучение TF-IDF модели спама с ручными признаками (Optuna). |
| `scripts/spam/analyze_feature_correlation.py` | Анализ корреляций признаков спама. |
| `scripts/spam/fix_labels_and_split.py` | Исправление разметки и разбиение датасета на train/val/test. |

### Примеры (обучение спама)

```bash
# Обучение TF-IDF спам-модели
python scripts/spam/train_spam.py \
  --train-data spam_data/train.parquet \
  --val-data spam_data/val.parquet \
  --output-dir models/spam/tfidf
```

---

## Запуск сервисов

| Скрипт | Назначение |
|--------|------------|
| `scripts/run/run_worker.py` | Воркер классификации: потребляет очередь запросов (RabbitMQ), запускает токсичность + спам, пишет в Postgres/Redis, публикует в очередь результатов. |
| `scripts/run/run_listener_telegram.py` | Listener для Telegram: приём сообщений, проверка кэша, публикация задач в очередь запросов. |
| `scripts/run/run_actions_telegram.py` | Action-сервис Telegram: потребляет очередь результатов, выполняет действия (предупреждение, удаление и т.д.) по правилам площадки. |

Перед запуском задайте переменные окружения (см. `.env.example`): `RABBITMQ_URL`, `DATABASE_URL`, `REDIS_URL` (при использовании кэша), для Telegram — токен бота и т.д.

### Примеры

```bash
# Воркер классификации (нужны RABBITMQ_URL, DATABASE_URL)
python scripts/run/run_worker.py

# Listener Telegram (нужны настройки Telegram и RabbitMQ)
python scripts/run/run_listener_telegram.py

# Action-сервис Telegram
python scripts/run/run_actions_telegram.py
```

---

## Healthcheck

| Скрипт | Назначение |
|--------|------------|
| `scripts/run/healthcheck_rabbitmq.py` | Проверка доступности RabbitMQ. |
| `scripts/run/healthcheck_worker.py` | Проверка готовности воркера (подключение к очередям, БД и т.д.). |

### Примеры

```bash
python scripts/run/healthcheck_rabbitmq.py
python scripts/run/healthcheck_worker.py
```

---

## Валидация цепочки и кэш

| Скрипт | Назначение |
|--------|------------|
| `scripts/run/validate_chain.py` | Валидация цепочки API → очередь → бэкенд → Postgres/кэш: загрузка текстов из parquet, батч-отправка POST /classify/batch-async, опрос GET /tasks/{task_id} до завершения; опционально повторная отправка части сообщений для проверки кэша. |
| `scripts/run/clear_redis.py` | Очистка Redis (кэш модерации). Использует `REDIS_URL` из окружения или `.env`. |

### Примеры

```bash
# Валидация цепочки (нужны API, воркер, RabbitMQ, Postgres, при опции --clear-cache — Redis)
python scripts/run/validate_chain.py --val-data data/val.parquet --max-samples 200
python scripts/run/validate_chain.py --val-data data/val.parquet -n 500 --duplicate-ratio 0.2 --clear-cache

# Очистка кэша Redis (REDIS_URL должен быть задан, например redis://localhost:6379/0)
python scripts/run/clear_redis.py
```

---

## Вспомогательные скрипты

| Скрипт | Назначение |
|--------|------------|
| `scripts/shared/cli.py` | Общие аргументы CLI для скриптов обучения (`--train-data`, `--val-data`, `--output-dir`, `--random-state`, Optuna и др.). Не запускается напрямую. |
| `scripts/shared/common.py` | Общие функции (подбор порога, loss и т.д.). |
| `scripts/shared/data.py` | Загрузка и подготовка данных (train/val, препроцессинг для токсичности/спама). |
| `scripts/augment_data.py` | Аугментация датасета для обучения. |

Пример запуска аугментации (параметры уточняйте в коде скрипта):

```bash
python scripts/augment_data.py
```

---

## Зависимости

- Для обучения BERT/RNN: PyTorch, transformers, зависимости из `requirements-api.txt` и скриптов.
- Для Optuna (train_tfidf, train_spam): `pip install optuna`.
- Для валидации спама (tfidf) с графиками: `pip install matplotlib`.
- Для воркера и listener/actions: RabbitMQ, Postgres, при необходимости Redis; переменные окружения из `.env.example`.
