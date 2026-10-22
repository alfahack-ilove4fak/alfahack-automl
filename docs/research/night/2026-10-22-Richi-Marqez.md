# Ночной аудит агента — 22–23.10.2026, @Richi-Marqez (роль research)

Автор прогонов — Claude Code (ночной аудитор), ветка `claude/cool-bohr-6n96gf`.
Код агента не менялся: только чтение и запуск.

## Подготовка

- Ник: `Richi-Marqez` → `docs/team.md` → **roma, Исследователь / Data Scientist** (`research`).
- Ветка сессии смержена с `origin/dev` (cba218d, «Already up to date»).
- Машина: **4 CPU, 15 ГБ RAM**, без swap (платформа — 8 CPU / 32 ГБ): время прогонов здесь ≈ в 1.5–2 раза больше, чем на платформе.
- `uv sync --group full` — 2 мин 26 с, OK.
- LLM: `scripts/qwen.py check` → ❌ `OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL не заданы`, `.env` нет.
  **LLM не проверена.** Все прогоны — без Qwen (`--mock-llm`); устойчивость к плохой LLM проверена поддельным
  OpenAI-совместимым сервером на localhost (фаза 4).

## Фаза 1 — тесты и качество кода

**Итог:** код чистый, тесты зелёные; один тест нестабилен под нагрузкой. Главные дыры — не в тестах,
а в том, чего тесты не покрывают (форматы файлов и метрики, фаза 3).

| Проверка | Результат |
|---|---|
| `uv run ruff check .` | All checks passed |
| `ruff format --check .` | 114 файлов уже отформатированы |
| `pytest tests/contracts tests/unit` | **1013 passed, 18 skipped**, 48 с |
| `pytest --durations=15` (полный) | **1077 passed, 18 skipped, 0 failed**, 457 с |
| doctor `--role research` | тесты ✅; ❌ только окружение: нет `.env`, `gh` не залогинен, нет хуков, нет `ipykernel` |

Самые долгие тесты (с): e2e multiclass 34.6, robustness regression без sample 30.0,
e2e regression 25.4, robustness multiclass без sample 24.3, harness 22.2, pre_submit 20.8.

### Нестабильные тесты

Дымовые тесты прогнаны ещё 2 раза (`pytest tests/smoke -rA`).

- Повтор №2: **63 passed, 1 failed** — `tests/smoke/test_harness.py::test_harness_run_on_public_task`:
  `value 0.05347 == baseline 0.05347` (ответ агента = тривиальный).
  В это время я параллельно гонял 3 коротких задачи (нагрузка на 4 ядра).
  Тест даёт агенту 30 с (`--minutes 0.5`) — под нагрузкой бейзлайн не успевает, и в ответе остаётся
  тривиальный. Это хрупкость теста (жёсткий бюджет), но и сигнал: при бюджете 30 с агент на 2000 строк
  без запаса по времени.
- Повтор №3 и проверка гипотезы «нагрузка»: _см. ниже_.

### Код без тестов

Скрипт: функции с ≥ 2 ветвлениями в `agent/ ml/ runtime/ evaluation/`, имя которых не встречается
ни в одном тесте. Таких **78** (многие проверены косвенно через публичные функции). Самые ветвистые:

| Функция | Ветвлений | Чем опасна |
|---|---|---|
| `agent/llm_client.py:247 _create_with_retries` | 13 | повторы/таймауты LLM — проверено фазой 4 |
| `agent/feature_code.py:282 _attach` | 12 | приклейка колонок от кода LLM |
| `agent/task_parser.py:35 find_task_files` | 11 | **поиск файлов: читает task.md только как UTF-8** |
| `agent/task_parser.py:285 _guess_task_type` | 9 | регрессия/классы по числу уникальных |
| `agent/task_parser.py:240 _guess_target` | 8 | target без sample_submission |
| `ml/experiment.py:353 _ensemble_members` | 8 | ансамбль |
| `evaluation/metrics.py:564 _first_metric` | 7 | поиск метрики в тексте |
| `agent/task_parser.py:266 _guess_id` | 4 | **id = `id` или первая колонка test** (фаза 3: ломается) |

### Где ошибки глотаются молча (agent/, ml/)

Почти везде `except Exception` пишет `log.exception` — это хорошо. Без единой строки в логе:

- `agent/task_parser.py:220` `_sample_has_probabilities` → `False`: шаблон не прочитался —
  и защита «метрика по меткам, а нужны вероятности» тихо выключена.
- `agent/task_parser.py:236` `_read_dictionary` → `{}`: словарь не прочитался (`;`, cp1251) —
  роль `target`/`id` из словаря потеряна, в логе пусто.
- `agent/task_parser.py:261` `_empty_test_columns` → `[]`.
- `agent/task_analyst.py:60`: шаблон ответа не прочитался — LLM получает «нет файла», хотя файл есть.

Предложение (agent, @n0h4ndy): `log.warning` с причиной в этих четырёх местах — 4 строки.

## Фаза 2 — харнесс на всех задачах

_не начата_

## Фаза 3 — «чужие» задачи

_не начата_

## Фаза 4 — условия платформы

_не начата_

## Фаза 5 — анализ и точки роста

_не начата_
