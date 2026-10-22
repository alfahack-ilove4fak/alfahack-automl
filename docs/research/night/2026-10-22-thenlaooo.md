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

- Пропуски (30): `importorskip` catboost/xgboost (группа full стоит — пропускаются ветки «не метрика/не регрессия» и `ML_BIG_TEST`), POSIX-сигналы и RLIMIT_AS — только `skipif`. Не расследовал каждый.
- Нестабильных тестов нет (3 прогона smoke, 0 расхождений). Самые долгие: end_to_end[multiclass] 35 с, harness 32 с, robustness ≈ 17 с каждый.
- ⚠️ `pyproject.toml: addopts="-q"` + `-q` в командах = `-qq` → итоговой строки «N passed» нет. Для сводки используйте `-o addopts=""`.
- ⚠️ `scripts/doctor.py --role captain` зовёт полный `pytest -x tests` и не уложился в 3 мин, без вывода — я убил его (дубль фазы 1). Для новичка «зелёный doctor» занимает > 3 мин на 4 ядрах без прогресс-бара.
- **Функции с ветвлениями ≥ 3 без упоминания в tests/** (эвристика по имени, приватные могут проверяться косвенно): `task_parser.find_task_files/_guess_target/_guess_id/_guess_task_type/_primary_target`, `llm_client._create_with_retries/_collect_stream/_complete_objects`, `ml/experiment._fit_catboost/_fit_xgboost/_ensemble_members/_fill_missing_categories`, `ml/tuning._space`, `runtime/watchdog._watch/_fire/_install_signals`, `evaluation/harness.holdout_mask`, `evaluation/metrics._first_metric/_find_words`. Парсер задачи (самое опасное для скрытых задач) тестируется только через `parse_task` целиком.
- **Молчаливые `except`** в agent/ и ml/: `agent/hypotheses.py:66` (`return {}` без лога), `agent/task_parser.py:220/236/260` (`_sample_has_probabilities`, `_read_dictionary`, `_empty_test_columns` — вернут «не нашли» без записи в лог; битый словарь = тихо нет roles), `agent/task_analyst.py:60`, `agent/critic.py:98`. Остальные `except Exception` пишут `log.exception`. Ничего критичного, но `_read_dictionary` при нестандартном разделителе скроет, что словарь не прочитан.

## Фаза 2 — харнесс
(не начата)

## Фаза 3 — «чужие» задачи
(не начата)

## Фаза 4 — условия платформы
(не начата)

## Фаза 5 — анализ и точки роста
(не начата)
