# Бенчмарки ModGuard: методика и хранилище результатов

Каталог содержит методику корректных замеров throughput/latency для полного
стенда (API → RabbitMQ → worker → PostgreSQL/Redis) и автоматически сохраняемые
результаты прогонов.

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
| Batch e2e latency | `completed_at − created_at` из Postgres (один источник времени, без квантования). |
| Intended e2e latency | `completed_epoch − (scheduled_arrival + clock_offset)` — честный e2e под нагрузкой, считается от **запланированного** времени прибытия (убирает coordinated omission). p50/p95/p99/p99.9/max. |
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
- `summary.csv` — по строке на прогон со стабильным набором колонок (для сравнения).
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
