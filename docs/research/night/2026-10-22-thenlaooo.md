# Ночной аудит 2026-10-22 (@thenlaooo, роль: капитан)

Ветка: `ccr-8b751caa-e44dvb`. Контейнер: 4 ядра, 15 ГБ ОЗУ (платформа: 8 / 32) — времена и RAM не равны боевым.

## Подготовка
- Роль: get_me → `thenlaooo` → капитан (docs/team.md).
- `git merge origin/dev`: Already up to date.
- `uv sync --group full`: OK (8 с).

## Фаза 1 — тесты и качество кода
**Итог:** ruff чистый; тесты 1065 прошло / 30 пропущено / 0 упало (первый полный прогон); smoke ещё 2 раза — 64/64 оба, расхождений нет.

| Что | Команда | Результат |
|---|---|---|
| Линтер | `uv run ruff check .` | All checks passed |
| contracts + unit | `uv run pytest tests/contracts tests/unit` | без падений (≈ 90 с) |
| Полный | `uv run pytest --durations=15` | 1065 ok, 30 skip, 0 fail |
| smoke ×2 | `uv run pytest tests/smoke -o addopts=""` | 64 passed за 361 с и 368 с |

- ⚠️ **Поправка (обнаружена в фазе 2):** в первом полном прогоне xgboost и catboost в окружении **не было** — `uv sync --group full` не смог скачать torch (хост `download-r2.pytorch.org` закрыт прокси контейнера), а `uv run` при `default-groups = ["dev"]` приводит окружение к группе dev и сносит остальное. Поэтому 30 пропусков — это тесты xgboost/catboost. После `uv pip install xgboost==2.1.4 catboost==1.2.8` и запуска через `.venv/bin/python`: contracts+unit **1013 passed, 18 skipped, 0 failed** (47 с); оставшиеся пропуски — `ML_BIG_TEST` (1) и «регрессии классы не нужны» (17), оба по делу. Smoke (64) прогнан без xgboost/catboost — с ними не перепроверен.
- Новичку: после `uv sync --group full` команда `uv run …` тихо удаляет тяжёлые пакеты. Нужно `uv run --group full` или `--no-sync`/`.venv/bin/python`.
- Нестабильных тестов нет (3 прогона smoke, 0 расхождений). Самые долгие: end_to_end[multiclass] 35 с, harness 32 с, robustness ≈ 17 с каждый.
- ⚠️ `pyproject.toml: addopts="-q"` + `-q` в командах = `-qq` → итоговой строки «N passed» нет. Для сводки используйте `-o addopts=""`.
- ⚠️ `scripts/doctor.py --role captain` зовёт полный `pytest -x tests` и не уложился в 3 мин, без вывода — я убил его (дубль фазы 1). Для новичка «зелёный doctor» занимает > 3 мин на 4 ядрах без прогресс-бара.
- **Функции с ветвлениями ≥ 3 без упоминания в tests/** (эвристика по имени, приватные могут проверяться косвенно): `task_parser.find_task_files/_guess_target/_guess_id/_guess_task_type/_primary_target`, `llm_client._create_with_retries/_collect_stream/_complete_objects`, `ml/experiment._fit_catboost/_fit_xgboost/_ensemble_members/_fill_missing_categories`, `ml/tuning._space`, `runtime/watchdog._watch/_fire/_install_signals`, `evaluation/harness.holdout_mask`, `evaluation/metrics._first_metric/_find_words`. Парсер задачи (самое опасное для скрытых задач) тестируется только через `parse_task` целиком.
- **Молчаливые `except`** в agent/ и ml/: `agent/hypotheses.py:66` (`return {}` без лога), `agent/task_parser.py:220/236/260` (`_sample_has_probabilities`, `_read_dictionary`, `_empty_test_columns` — вернут «не нашли» без записи в лог; битый словарь = тихо нет roles), `agent/task_analyst.py:60`, `agent/critic.py:98`. Остальные `except Exception` пишут `log.exception`. Ничего критичного, но `_read_dictionary` при нестандартном разделителе скроет, что словарь не прочитан.

## Фаза 2 — харнесс на всех задачах
Условия: контейнер 4 ядра / 15 ГБ, **LLM не проверена** (нет ключа) → все прогоны без LLM. Харнесс не знает `--mock-llm`: без `--llm` LLM и так выключена. Потоки шли параллельно, каждый на 2 ядрах (`taskset`), поэтому времена ≈ ×2 к 4 ядрам и тем более к 8 ядрам платформы.

**Харнесс `--all --minutes 5`** (с xgboost/catboost, 2 ядра, commit 654a66b), балл к сильному бейзлайну:

| Задача | Метрика | Агент | Сильный бейзлайн | Балл (сильн.) |
|---|---|---|---|---|
| multiclass | f1_macro | 0.5503 | 0.5144 | 7.4 |
| regression | rmse | 50.41 | 56.80 | 11.3 |
| claims (MAPE) | mape | 0.4889 | 0.5912 | 17.3 |
| cancer | roc_auc | 0.9929 | 0.9879 | 41.4 |
| digits | accuracy | 0.9873 | 0.9714 | 55.6 |
| payments | f1 | 0.3186 | 0.25 | 9.2 |
| sales | rmsle | 0.4699 | 0.4831 | 2.7 |
| network | logloss | 0.1054 | 0.1076 | 2.0 |
| batches | qwk | 0.7259 | 0.7243 | 0.6 |
| **diabetes** | mae | 48.98 | 47.60 | **0.0** |

Среднее к сильному **14.7**; 10/10 OK; время агента 159–264 с из 300; пик RAM 223–377 МБ.

- Тот же харнесс на 4 ядрах без xgboost/catboost (до моей поправки окружения): среднее **17.0**; multiclass F1 **0.6457** (балл 27.0), payments 0.332. Разница почти вся на multiclass: за 5 минут на 2 ядрах агент не дошёл до `balanced_proba_weights`. Вывод: в 5-минутном харнессе порядок playbook и скорость машины решают больше, чем «ум» агента; в бою (180 мин, 8 ядер) этот эффект исчезает, но сравнивать прогоны между машинами нельзя.
- **diabetes 0.0** — тот же шум, что в #206 (231 строка, MAE 48.98 против 47.60 у LightGBM по умолчанию). Подгонять не стоит.
- Задачи с баллом < 3 к сильному (batches 0.6, network 2.0, sales 2.7, diabetes 0.0) — агент ≈ LightGBM по умолчанию: playbook на них ничего не даёт.

**`--minutes 15` на трёх слабых** (batches, diabetes, network; 2 ядра): ответ **идентичен** 5-минутному до 5-го знака (qwk 0.72593; mae 48.98; logloss 0.10544); время 733 / 669 / 733 с из 900. 10 лишних минут = 0 баллов. Агент работал около 4 минут, остальное — Optuna и запасные гипотезы, не принятые правилом t ≥ 2.

**LLM против без LLM (`scripts/qwen.py compare`) — не запускал: нет `OPENAI_API_KEY`/`OPENAI_MODEL`.** Про токены и ₽ — см. фазу 4 (мок-LLM, только устойчивость, не качество).

**Прогон на 60 мин (regression, 2 ядра, без LLM, `scripts/run_local.py --minutes 60 --mock-llm`):**
- Агент закончил за **601 с из 3600 (17%)**: «Гипотезы закончились — завершаю цикл». 19 экспериментов.
- Хронология: бейзлайн к 28 с (RMSE 49.388 ± 4.11); playbook 13 гипотез до 260 с; единственная принятая — `freq_encoding` (49.388 → 48.596, t=2.36, 4 фолда); запасные (#200) xgboost 48.82, catboost 49.61, hist_gb 53.62, Optuna 48.24 (293 с), комбинация 48.72 — ни одна не прошла порог. 
- Итого запасные направления из #200 уже есть (fallback.py), но их **5 штук**, и они тоже кончаются за ≈ 5 минут. В бою на 8 ядрах и 180 мин простаивает ≈ **165–170 мин из 180 (≈ 93%)**; оценка по этому прогону: 10 мин работы из 60 → тот же набор гипотез займёт на 8 ядрах 5–8 мин.
- Принято `freq_encoding` при t=2.36 на 4 фолдах после проверки ≈ 18 гипотез: при пороге t ≥ 2 на каждую ожидаемо ~1 ложное принятие на 20 — риск «проклятия победителя» (в журнале на этой же задаче `freq_encoding` раньше давал t=0.79).

## Фаза 3 — «чужие» задачи
(не начата)

## Фаза 4 — условия платформы
(не начата)

## Фаза 5 — анализ и точки роста
(не начата)
