# Ночной аудит 2026-10-22 (thenlaooo, роль: капитан)

Ветка: `ccr-3d8d3c07-vujae9`. Аудитор — Claude (автономно). Отчёт дописывается после каждой фазы.

## Подготовка
- Пользователь: GitHub `thenlaooo` → docs/team.md → vasya, капитан/интегратор.
- Контейнер: 4 CPU / 15 ГБ ОЗУ (платформа: 8 CPU / 32 ГБ) — времена и RAM ниже нужно умножать на ~2 в худшую сторону при выводах о бюджете; параллельные прогоны искажают время.
- origin/dev уже в ветке (cba218d), merge ничего не добавил.
- `uv sync --group full` **не прошёл** (нет доступа к download.pytorch.org: tunnel error) → torch/autogluon/lightautoml/langchain не ставились. Вручную поставлены: catboost, xgboost, polars, pyarrow, imbalanced-learn, category-encoders, feature-engine, statsmodels, flaml.
- LLM: из контейнера сети до Qwen нет (DNS/403 прокси сессии). Облачный `qwen check` (Actions #4): ✅ связь, 513.27 ₽ из 600 ₽ осталось на 20:52 UTC. Все локальные прогоны — `--mock-llm`; **LLM локально не проверена**. Прогоны с реальной Qwen — только через Actions (workflow `qwen`).

## Фаза 1 — тесты и качество кода (готово)

| Проверка | Результат |
|---|---|
| `ruff check .` | ✅ All checks passed |
| `pytest` целиком (`--durations=15`) | ✅ **1077 passed, 18 skipped, 0 failed** (~17 мин на 4 CPU) |
| smoke (`tests/smoke`, 64 теста) ещё 2 раза | ✅ 64/64 и 64/64 — **нестабильных нет** |
| unit+contracts с покрытием по ветвлениям | 91% (3388 строк, 1152 ветви) |

Пропуски (18): 17 — параметризация `test_metrics.py:47` «у регрессии нет классов» (штатно), 1 — `test_big_synthetic_fits_budget`
(1.5 млн строк, нужен `ML_BIG_TEST=1`). **Этот тест не запускается нигде** (ни CI, ни `pre_submit`) — единственная проверка
масштаба 1M+ строк. Мой прогон на 1M — в фазе 3 (`big_1m_rows`).

`doctor.py --role captain` без `--quick` за 150 с не вывел ничего (отчёт печатается только в конце, дымовые тесты внутри);
с `--quick`: 1013 passed, 18 skipped за 66 с. ❌-пункты в контейнере ожидаемы (нет `.env`, хука, входа `gh`).

### Покрытие: самые слабые места (unit+contracts, ветви)
| Модуль | Покрытие | Что не покрыто |
|---|---|---|
| `evaluation/harness.py` | 51% | почти всё тело `run_one` (проверяется только smoke) |
| `runtime/log.py` | 84% | строки 29–32 (запись в файл) |
| `runtime/sandbox.py` | 85% | ветки ошибок запуска (`exit_code=127`), Windows |
| `runtime/watchdog.py` | 87% | аварийные ветки (`_restore_signals`, финализация при исключении) |
| `ml/validation.py` | 88% | `group:<колонка>` (158–161), `time` без колонки (164–167), ветки отбора кандидатов |
| `evaluation/submission.py` | 89% | чтение CSV с ошибкой, лишние/пропущенные id |
| `ml/tuning.py` | 89% | Optuna: пустые/вырожденные пространства |

### Проглоченные ошибки (`except Exception` — 26 мест в agent/ ml/ runtime/ evaluation/ solution.py)
Почти везде есть `log.exception`/`log.warning` — **молчаливо глотаются только 5**, все малозначимы, но невидимы в логе:
- `agent/task_analyst.py:60` — не прочитался `sample_submission` для подсказки LLM (`pass`);
- `agent/task_parser.py:220` (`_sample_has_probabilities`), `:236` (`_read_dictionary`), `:261` (`_empty_test_columns`) —
  возвращают «нет» без записи в лог. **`_read_dictionary` самый неприятный:** сломанный словарь данных → роль `target`
  теряется молча, и target может быть угадан иначе;
- `agent/hypotheses.py:66` — `_model_defaults()` без `ml`.
Предложение: одна строка `log.info(...)` в каждом (владелец `agent/`).

### Найдено чтением кода (подтверждается в фазах 3–4)
- Нигде в `agent/ ml/ evaluation/ solution.py` нет разбора разделителя/кодировки CSV: `pd.read_csv(path)` по умолчанию
  (`agent/task_parser.py:67-68,74,108`, `ml/profiling.py:19-20`, `solution.py:164-170`).
- `agent/feature_code.py`: AST-фильтр кода LLM пропускает `pd.core.common.builtins.open(...)` (проверено `check_feature_code`:
  `None` = «допустим») — `open` проверяется только как имя, а не как атрибут. Песочница (`runtime/sandbox.py`) файлы не
  изолирует (сказано в её docstring). Для своего Qwen без сети — низкий риск, но «запрет файлов» держится не везде.
- `ml/validation.py:72-74` `time_values` вызывает `pd.to_datetime(строки, errors="coerce")` для каждой строковой колонки, train
  и test, в `choose_validation`: на 200 тыс. уникальных строк (хэш/текст) **9–10 с на колонку** (замер при загруженном
  CPU) → на 1M строк это минуты бюджета бейзлайна. См. фазу 3, `big_1m_rows`.
- `agent/task_parser.py:88` + `agent/task_analyst.py` `reconcile`: если метрика *найдена поиском по тексту*, LLM её не
  заменит, даже если ответила иначе («оставлен поиск» — только заметка в notes). Проверено на фразах, см. фазу 3.

## Фаза 2 — харнесс на всех задачах
_не начата_

## Фаза 3 — «чужие» задачи
_не начата_

## Фаза 4 — условия платформы
_не начата_

## Фаза 5 — анализ и точки роста
_не начата_
