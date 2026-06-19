# Канонические результаты throughput / latency (2026-06-19)

Замеры выполнены на полном стенде **API → RabbitMQ → 4× backend worker → PostgreSQL/Redis**
после оптимизаций classify (preprocess tox, TF-IDF spam/tox `char_wb (3,4)`).

**Git commit:** `e504603`  
**Датасет:** `data/toxicity/val.parquet`, 65 115 текстов  
**Модели:** `tox_model=regex+tfidf`, `spam_model=regex+tfidf` (обе TF-IDF `ngram (3,4)`)

## Условия стенда

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

---

## 1. Пиковая throughput (capacity)

**Режим:** closed-loop дамп — все 66 батчей отправляются сразу, без контроля темпа  
(`--target-rate 0`, `--delay-min-ms 0 --delay-max-ms 0`)

| Параметр | Значение |
|----------|----------|
| `batch_size` | **1000** |
| `send_workers` | 16 |
| `poll_interval` | 0.1 s |
| Повторов | **3** (`capacity-r1…r3`) |

### Результаты (медиана 3 прогонов)

| Метрика | Значение | Примечание |
|---------|----------|------------|
| **Throughput steady (PG, trim=10%)** | **≈ 15 878 msg/s** | **главное число «потолка»** |
| Диапазон steady | 14 546 – 16 084 msg/s | разброс между прогонами |
| Throughput overall (PG) | ≈ 15 000 msg/s | вкл. разгон/затухание |
| Success rate | 100% | 68/68 task_id, 65 115 items |
| Cache hit | ~0% | cold run |

### Latency при capacity (не путать с latency-under-load)

При дампе система перегружена; batch e2e **занижает** очередь (coordinated omission).

| Метрика | p50 | p95 | p99 |
|---------|-----|-----|-----|
| Batch e2e (PG) | ~1.7 s | ~2.6 s | ~2.9 s |
| **Intended e2e** (от scheduled arrival) | **~2.6 s** | **~4.4 s** | **~4.6 s** |

Файлы: `benchmarks/results/run_20260619T151112Z_capacity-r{1,2,3}.json`

---

## 2. Latency одиночного запроса (single-message)

**Режим:** последовательные POST `/classify` по одному тексту, cold/hot cache  
(`--single-latency-n 300`, `--single-latency-mode cold-hot`, `poll_interval=0.05`)

| Метрика | Cold cache | Hot cache |
|---------|------------|-----------|
| **Server-side p50 (PG)** | **7 ms** | **1 ms** |
| Server-side p95 | 9 ms | 2 ms |
| Server-side p99 | 10 ms | 3 ms |
| Client p50 (с poll) | 64 ms | 6 ms |

Server-side = `completed_at − created_at` в Postgres, без квантования poll.  
Client-side включает `poll_interval=50 ms` → cold ≈ 1 пауза опроса + обработка.

---

## 3. Latency under load (open-loop)

**Режим:** запросы прибывают по **Пуассоновскому** процессу с фиксированным λ  
(`--arrival poisson`, `--batch-size 50`, `load_n=3000`, 2 повтора на точку)

Capacity **C = 15 800 msg/s** (медиана capacity-прогонов).

| Доля от C | Offered λ, msg/s | Achieved, msg/s | Intended p50 | Intended p95 | Intended p99 |
|-----------|------------------|-----------------|--------------|--------------|--------------|
| 30% | 4 740 | ≈ 4 814 | **25 ms** | 62 ms | 82 ms |
| 50% | 7 900 | ≈ 8 326 | **26 ms** | 41 ms | 50 ms |
| 70% | 11 060 | ≈ 10 694 | **38 ms** | 61 ms | 70 ms |
| 85% | 13 430 | ≈ 9 582 | **98 ms** | 149 ms | 178 ms |

**«Колено» насыщения:** при **~85% C** achieved падает ниже offered (очередь растёт), p99 ≈ 178 ms.  
Рабочая зона с p99 < 100 ms: **до ~70% C** (λ ≈ 11k msg/s).

Файлы: `benchmarks/results/sweep_20260619T151350Z.{csv,md}`

---

## Команды воспроизведения

### Capacity (3 повтора)

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

### Single latency

```bash
.venv/bin/python scripts/run/validate_chain.py \
  --val-data data/toxicity/val.parquet \
  --single-latency-n 300 --single-latency-mode cold-hot \
  --poll-interval 0.05 --clear-cache --duplicate-ratio 0 \
  --run-tag single-latency \
  --backend-replicas 4 --pool-workers 1 --prefetch 12
```

### Latency-vs-load sweep

```bash
.venv/bin/python scripts/run/benchmark_sweep.py \
  --val-data data/toxicity/val.parquet \
  --batch-size 50 --capacity 15800 \
  --load-n 3000 --fracs 0.3,0.5,0.7,0.85 --repeat 2 \
  --arrival poisson --send-workers 16 \
  --backend-replicas 4 --pool-workers 1 --prefetch 12
```

---

## Краткие выводы для диссертации

1. **Пиковая пропускная способность:** **~15.9k msg/s** (steady-state PG, 4 worker, batch=1000).
2. **Latency одиночного запроса (cold):** **7 ms** server-side (tox+spam, both models).
3. **Latency под нагрузкой:** p50 **25–40 ms** до 70% C; при 85% C — деградация до p99 **178 ms**.
4. Capacity batch e2e **~2 s** — метрика перегрузки, не SLA; для SLA использовать open-loop intended latency.
