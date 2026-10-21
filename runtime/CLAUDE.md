# runtime/ и solution.py — роль: капитан / интегратор (кто сейчас — docs/team.md)

Пакет роли: @docs/roles/captain.md · Материалы: docs/resources/captain.md

## Что здесь
| Файл | Роль |
|---|---|
| `../solution.py` | точка входа платформы: склеивает модули, гарантирует ответ на каждом шаге |
| `budget.py` | `RunBudget` из переменных окружения, доступная память (с учётом лимита Docker) |
| `log.py` | единый лог: stderr + `runs/.../agent.log`, время от старта в каждой строке |
| `sandbox.py` | `run_code(code, workdir, timeout, mem_mb) -> CodeResult`: код от LLM в отдельном процессе, таймаут, лимит памяти (Linux, mem_mb ≥ 2048 для pandas), чистое окружение |

## Правила
- `solution.py` — тонкий: логика — в модулях. Интерфейс строго как в `reference/baseline.py` (argv[1] = путь к task.md, env `OPENAI_*`).
- Любое изменение `solution.py` проверяется сквозным тестом `tests/smoke` и прогоном `scripts/run_local.py --all`.
- Запас по времени (`safety_seconds`) не трогать без замера длительности финального обучения на полных данных.

## Стыкуется с
все модули (интеграция) · qa (сабмиты, CI) · agent (цикл и бюджет).
