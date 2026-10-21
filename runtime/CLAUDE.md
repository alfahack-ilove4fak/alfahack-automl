# runtime/ и solution.py — роль: капитан / интегратор (кто сейчас — docs/team.md)

Пакет роли: @docs/roles/captain.md · Материалы: docs/resources/captain.md

## Что здесь
| Файл | Роль |
|---|---|
| `../solution.py` | точка входа платформы: склеивает модули, гарантирует ответ на каждом шаге |
| `budget.py` | `RunBudget` из переменных окружения, доли времени по этапам (`plan_stages`), память (с учётом лимита Docker) |
| `watchdog.py` | сторож: жёсткий дедлайн и сигналы SIGTERM/SIGINT → проверить `submission.csv` и завершить процесс |
| `log.py` | единый лог: stderr + `runs/.../agent.log`, время от старта в каждой строке |
| (план) `sandbox.py` | запуск кода от LLM в отдельном процессе с таймаутом и лимитом памяти |

## Правила
- `solution.py` — тонкий: логика — в модулях. Интерфейс строго как в `reference/baseline.py` (argv[1] = путь к task.md, env `OPENAI_*`).
- Любое изменение `solution.py` проверяется сквозным тестом `tests/smoke` и прогоном `scripts/run_local.py --all`.
- Запас по времени (`safety_seconds`) не трогать без замера длительности финального обучения на полных данных.

## Стыкуется с
все модули (интеграция) · qa (сабмиты, CI) · agent (цикл и бюджет).

## Время по этапам и сторож (C-L2-2, C-L3-1)
- Доли от `AGENT_TIME_BUDGET_SEC`: разбор + бейзлайн ≤ 20% (мягко: только предупреждение в логе), AutoResearch — остаток (65%),
  финал ≥ 10%, запас до лимита платформы ≥ 5%. Константы `BASELINE_SHARE / FINAL_SHARE / SAFETY_SHARE` в `budget.py`,
  на запуске меняются `AGENT_BASELINE_SHARE / AGENT_FINAL_SHARE / AGENT_SAFETY_SHARE`; `AGENT_SAFETY_SEC` задаёт резерв в секундах.
- `RunBudget.safety_seconds` = финал + запас: `has_time()` в цикле и в фолдах CV обрывает AutoResearch за него до конца.
  После бейзлайна `solution.py` замеряет финальное обучение и через `ensure_final_reserve` увеличивает резерв
  (≥ запас + 3 × замер), если финал тяжёлый. Резерв только растёт.
- Сторож срабатывает за `SAFETY_SHARE` бюджета до лимита платформы (на 3 часах — за 9 минут) или по сигналу: ждёт конца
  записи ответа (`keeper.lock`), проверяет `submission.csv`, при порче возвращает последний валидный ответ
  (нет даже его — образец `sample_submission.csv`), делает `os._exit`. Код выхода: 0 по дедлайну, 128+сигнал по сигналу.
- Поток сторожа нужен, потому что LightGBM минутами не отдаёт управление питону: обычный `signal.signal` и проверки `has_time()` его не прервут.
- Новый код, который пишет `submission.csv`, обязан писать через `AnswerKeeper.write` (или хотя бы `write_submission_atomic`).
- Тесты: `tests/unit/test_budget.py`, `test_watchdog.py`, `test_answer_keeper.py`, сквозной `tests/smoke/test_killed_mid_loop.py`
  (зависание, SIGTERM, исключение в цикле).
