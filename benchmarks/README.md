# Бенчмарки ModGuard: методика и хранилище результатов

Каталог содержит методику корректных замеров throughput/latency для полного
стенда (API → RabbitMQ → worker → PostgreSQL/Redis) и автоматически сохраняемые
результаты прогонов.

## Актуальные результаты (2026-06-19)

Замеры на полном стенде **API → RabbitMQ → 4× backend worker → PostgreSQL/Redis**
после оптимизаций classify (preprocess tox, TF-IDF spam/tox `char_wb (3,4)`).

**Git commit:** `e504603`  
**Датасет:** `data/toxicity/val.parquet`, 65 115 текстов  
**Модели:** `tox_model=regex+tfidf`, `spam_model=regex+tfidf` (обе TF-IDF `ngram (3,4)`)

### Условия стенда

| Параметр | Значение |
|----------|----------|
| `docker compose scale backend` | **4** |
| `MODERATION_POOL_WORKERS` | **1** (на контейнер) |
| Worker prefetch | **12** |
| `MODERATION_PIPELINE` | `both` |
| API | `http://localhost:8000` |
| Кэш | **очищен** перед каждым прогоном (`--clear-cache`) |
| Дубликаты | **0** (`--duplicate-ratio 0`) |
| Метрики времени | **PostgreSQL** (`tasks.created_at` / `completed_at`) |

### 1. Пиковая throughput (capacity)

**Режим:** closed-loop дамп — все батчи отправляются сразу, без контроля темпа  
(`--target-rate 0`, `--delay-min-ms 0 --delay-max-ms 0`).

| Параметр | Значение |
|----------|----------|
| `batch_size` | **1000** |
| `send_workers` | 16 |
| `poll_interval` | 0.1 s |
| Повторов | **3** (`capacity-r1…r3`) |

**Результаты (медиана 3 прогонов):**

| Метрика | Значение | Примечание |
|---------|----------|------------|
| **Throughput steady (PG, trim=10%)** | **≈ 15 878 msg/s** | главное число «потолка» |
| Диапазон steady | 14 546 – 16 084 msg/s | разброс между прогонами |
| Throughput overall (PG) | ≈ 15 000 msg/s | вкл. разгон/затухание |
| Success rate | 100% | 68/68 task_id, 65 115 items |
| Cache hit | ~0% | cold run |

При capacity-дампе система перегружена; batch e2e **занижает** очередь
(coordinated omission). Для SLA используйте open-loop с prod-размером батча (раздел 4).

| Метрика | p50 | p95 | p99 |
|---------|-----|-----|-----|
| Batch e2e (PG) | ~1.7 s | ~2.6 s | ~2.9 s |
| Intended e2e (от scheduled arrival) | ~2.6 s | ~4.4 s | ~4.6 s |

Файлы: `benchmarks/results/run_20260619T151112Z_capacity-r{1,2,3}.json`

### 2. Latency одиночного запроса (single-message)

**Режим:** последовательные POST `/classify` по одному тексту, cold/hot cache  
(`--single-latency-n 300`, `--single-latency-mode cold-hot`, `poll_interval=0.05`).

| Метрика | Cold cache | Hot cache |
|---------|------------|-----------|
| **Server-side p50 (PG)** | **7 ms** | **1 ms** |
| Server-side p95 | 9 ms | 2 ms |
| Server-side p99 | 10 ms | 3 ms |
| Client p50 (с poll) | 64 ms | 6 ms |

Server-side = `completed_at − created_at` в Postgres, без квантования poll.  
Client-side включает `poll_interval=50 ms` → cold ≈ 1 пауза опроса + обработка.

### 3. Кривая latency-vs-load (batch=50, sweep)

**Режим:** запросы прибывают по **Пуассоновскому** процессу с фиксированным λ  
(`--arrival poisson`, `--batch-size 50`, `load_n=3000`, 2 повтора на точку).  
Используется для построения **кривой насыщения** (где система начинает отставать
от offered rate). Размер батча 50 — удобен для sweep; для prod-SLA см. раздел 4.

Capacity **C = 15 800 msg/s** (медиана capacity-прогонов).

| Доля от C | Offered λ, msg/s | Achieved, msg/s | Intended p50 | Intended p95 | Intended p99 |
|-----------|------------------|-----------------|--------------|--------------|--------------|
| 30% | 4 740 | ≈ 4 814 | **25 ms** | 62 ms | 82 ms |
| 50% | 7 900 | ≈ 8 326 | **26 ms** | 41 ms | 50 ms |
| 70% | 11 060 | ≈ 10 694 | **38 ms** | 61 ms | 70 ms |
| 85% | 13 430 | ≈ 9 582 | **98 ms** | 149 ms | 178 ms |

**Колено насыщения:** при **~85% C** achieved падает ниже offered (очередь растёт),
p99 ≈ 178 ms. Рабочая зона с intended p99 < 100 ms: **до ~70% C** (λ ≈ 11k msg/s).

Файлы: `benchmarks/results/sweep_20260619T151350Z.{csv,md}`

### 4. Prod-сценарий: batch=1000 @ 10k msg/s

**Режим:** open-loop, **Пуассон**, λ = **10 000 msg/s**, `batch_size=1000`
(≈10 батчей/с, ~63% от C). Соответствует планируемому prod: `MAX_BATCH_SIZE=1000`,
`POST /classify/batch-async`.

| Параметр | Значение |
|----------|----------|
| `target_rate` | **10 000 msg/s** |
| `batch_size` | **1000** |
| `n_items` | 50 000 (50 батчей) |
| Achieved throughput (PG) | **≈ 9 973 msg/s** |
| Success rate | 100% |

**Batch e2e** (`completed_at − created_at` в PG) — главная SLA-метрика для батча:

| Метрика | p50 | p95 | p99 |
|---------|-----|-----|-----|
| **Batch e2e** | **175 ms** | 205 ms | 242 ms |

**Intended e2e** (per-item, от запланированного прибытия до завершения батча):

| Метрика | p50 | p95 | p99 |
|---------|-----|-----|-----|
| Intended e2e | 248 ms | 311 ms | 343 ms |

Intended выше batch e2e: ранние сообщения в батче ждут наполнения (~100 ms при 10k/s)
и обработки всего батча.

Файл: `benchmarks/results/run_20260619T153030Z_load-10k-batch1000.json`

### Команды воспроизведения (канонический прогон)

Capacity (3 повтора):

```bash
for i in 1 2 3; do
  .venv/bin/python scripts/run/validate_chain.py \
    --val-data data/toxicity/val.parquet \
    --batch-size 1000 --batch-window 3 \
    --delay-min-ms 0 --delay-max-ms 0 \
    --duplicate-ratio 0 --send-workers 16 \
    --poll-interval 0.1 --clear-cache --label-col '' \
    --run-tag "capacity-r$i" \
    --backend-replicas 4 --pool-workers 1 --prefetch 12
done
```

Single latency:

```bash
.venv/bin/python scripts/run/validate_chain.py \
  --val-data data/toxicity/val.parquet \
  --single-latency-n 300 --single-latency-mode cold-hot \
  --poll-interval 0.05 --clear-cache --duplicate-ratio 0 \
  --run-tag single-latency \
  --backend-replicas 4 --pool-workers 1 --prefetch 12
```

Latency-vs-load sweep (batch=50, кривая насыщения):

```bash
.venv/bin/python scripts/run/benchmark_sweep.py \
  --val-data data/toxicity/val.parquet \
  --batch-size 50 --capacity 15800 \
  --load-n 3000 --fracs 0.3,0.5,0.7,0.85 --repeat 2 \
  --arrival poisson --send-workers 16 \
  --backend-replicas 4 --pool-workers 1 --prefetch 12
```

Prod-сценарий (batch=1000 @ 10k msg/s):

```bash
.venv/bin/python scripts/run/validate_chain.py \
  --val-data data/toxicity/val.parquet \
  -n 50000 \
  --batch-size 1000 --batch-window 3 \
  --target-rate 10000 --arrival poisson \
  --delay-min-ms 0 --delay-max-ms 0 \
  --duplicate-ratio 0 --send-workers 16 \
  --poll-interval 0.1 --clear-cache --label-col '' \
  --run-tag load-10k-batch1000 \
  --backend-replicas 4 --pool-workers 1 --prefetch 12
```

---

## Зачем отдельная методика

Наивный замер «отправить всё разом и поделить на время» (capacity / closed-loop
дамп) показывает только потолок системы при перегрузке и систематически занижает
latency из-за **coordinated omission**: пока система занята, новые запросы не
отправляются, поэтому их ожидание в очереди не учитывается. Для адекватной оценки
нужны два разных эксперимента:

- **Capacity (потолок)** — сколько сообщений в секунду система переваривает при
  перегрузке. Это дамп без контроля темпа (`--target-rate 0`, режим по умолчанию).
- **Latency-under-load** — какова задержка при фиксированном входном темпе
  (open-loop). Запросы прибывают по расписанию с интенсивностью λ независимо от
  того, успевает ли система, поэтому ожидание в очереди попадает в latency.

## Метрики

| Метрика | Что значит |
|---|---|
| `throughput_overall` | items / (max(completed_at) − min(created_at)) по всем задачам (PG). Включает разгон/затухание. |
| `throughput_steady` | то же, но на установившемся плато: первые и последние `--steady-trim` доли окна отброшены. Это «честный» потолок. |
| `achieved_throughput` | фактически доставленный темп (steady, иначе overall) — главное число для сравнения. |
| Batch e2e latency | `completed_at − created_at` из Postgres (один источник времени, без квантования). **Основная SLA-метрика** для prod при `batch-async` (время обработки целого батча). |
| Intended e2e latency | `completed_epoch − (scheduled_arrival + clock_offset)` — per-item e2e под нагрузкой, от **запланированного** прибытия (убирает coordinated omission). Включает ожидание наполнения батча. |
| Single-request (server-side) | `completed − created` из PG для одиночных запросов — без квантования `--poll-interval`. Клиентская latency с poll выводится отдельно как вторичная. |

`clock_offset` = NOW() Postgres − локальное время клиента — меряется один раз;
round-trip к БД мал, для e2e в сотни мс погрешность пренебрежима.

## Одиночный прогон

Capacity (потолок) с сохранением результата:

```bash
python scripts/run/validate_chain.py \
   --val-data data/toxicity/val.parquet \
   --batch-size 50 --batch-window 3 \
   --clear-cache \
   --run-tag capacity --backend-replicas 4 --pool-workers 1 --prefetch 12
```

Latency-under-load (open-loop, например 400 msg/sec, пуассоновский поток):

```bash
python scripts/run/validate_chain.py \
   --val-data data/toxicity/val.parquet \
   --batch-size 50 --batch-window 3 \
   --clear-cache --duplicate-ratio 0 \
   --target-rate 400 --arrival poisson \
   --run-tag load-400
```

Флаги `--backend-replicas/--pool-workers/--prefetch` ни на что не влияют в самом
прогоне — они лишь записываются в метаданные, чтобы зафиксировать условия стенда.
Отключить сохранение: `--no-save`. Сменить каталог: `--results-dir`.

## Профилирование по этапам (крупные батчи)

Capacity-дамп смешивает 33 параллельных батча — в chain_timing попадают healthcheck
и средние по очереди неинформативны. Для разбивки **POST / pg / classify / redis**
на одном большом батче:

```bash
CHAIN_TIMING=1 docker compose up -d --build --scale backend=4 api backend

python scripts/run/bench_stages.py \
   --val-data data/toxicity/val.parquet \
   --batch-size 2000 --repeat 5 --warmup 1 --clear-cache
```

Батчи идут **последовательно** (без очереди). Сводка этапов:

```bash
docker compose logs --since 5m api backend 2>&1 \
   | python scripts/run/summarize_chain_timing.py --min-items 500
```

`--min-items` отсекает healthcheck (n=1). В both-режиме отдельно: `classify_tox`,
`classify_spam`, `max(tox,spam)`.

## Кривая latency-vs-load (sweep)

```bash
python scripts/run/benchmark_sweep.py \
   --val-data data/toxicity/val.parquet \
   --batch-size 50 --batch-window 3 \
   --fracs 0.3,0.5,0.7,0.85,0.95 --repeat 3 \
   --backend-replicas 4 --pool-workers 1 --prefetch 12
```

Скрипт сначала измеряет потолок `C` (capacity-probe), затем гоняет open-loop при
λ = долях `C`, повторяя каждую точку `--repeat` раз и беря медиану. Итог —
таблица offered_rate / achieved_throughput / p50 / p95 / p99, где видно «колено»
насыщения. Можно задать потолок вручную: `--capacity <C>`.

## Файлы результатов (`results/`)

- `run_<UTC>_<tag>.json` — полный результат одного прогона (условия + метрики).
- `run_<UTC>_<tag>.md` — человекочитаемый отчёт того же прогона.
- `summary.csv` — по строке на прогон (локально, **не коммитится** — растёт при каждом запуске).
- `sweep_<UTC>.{csv,md}` — сводная кривая latency-vs-load.

### Схема summary.csv

`timestamp, git_commit, run_tag, mode, target_rate, achieved_throughput,
throughput_steady, throughput_overall, e2e_p50/p95/p99/p999/max,
intended_p50/p95/p99/p999/max, single_server_p50/p95/p99, success_rate,
cache_hit_rate, n_items, batch_size, batch_window, send_workers,
backend_replicas, pool_workers, prefetch, model_tox, model_spam, note`.

`mode` = `capacity` (дамп) либо `open-loop`. Модели (`model_tox`/`model_spam`)
берутся из PG как доминирующие `tox_model_used`/`spam_model_used` по task_items.

## Замечания

- Для intended/single серверной latency нужен `DATABASE_URL` (метрики читаются из PG).
- Per-item intended latency требует один task_id на батч, поэтому в latency-прогонах
  используйте `--clear-cache --duplicate-ratio 0` (частичный кэш даёт 2 task_id и
  такие батчи в intended-latency не учитываются).
- В open-loop при перегрузке POST-приём может частично связать клиентский цикл
  (backpressure) — это реалистично; фактический `achieved_throughput` фиксируется
  отдельно от offered rate.
