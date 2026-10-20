# agent/ — владелец: LLM/agent-инженер (@n0h4ndy)

Пакет роли: @docs/roles/agent.md · Материалы: docs/resources/agent.md

## Что здесь
| Файл | Роль |
|---|---|
| `task_parser.py` | `task.md` + файлы → `TaskSpec`. Детерминированный, работает без LLM — не ломать! |
| `llm_client.py` | Qwen через OpenAI-совместимый API: кэш на диск, учёт ₽, `available()` = False без ключа |
| `hypotheses.py` | откуда берутся гипотезы: сейчас только `research/playbook.yaml`, дальше — LLM |
| `loop.py` | цикл AutoResearch: гипотеза → `ml.run_experiment` → принять, если лучше на `min_delta` |

## Контракты
- Создаёшь: `TaskSpec`, `Hypothesis`. Читаешь: `DataProfile`, `ExperimentResult`, `RunBudget`.
- `Hypothesis.params` должен быть понятен `ml/experiment.py` — список ключей в его docstring. Новый ключ → договорись с ML-инженером (стыковка ежедневно в 13:00).

## Правила модуля
- **LLM — улучшение, а не опора.** Любой путь кода должен работать при `LLM_MOCK=1`. Ответ LLM всегда валидируй (pydantic), мусор → пропускаем гипотезу.
- Код, который пишет LLM, исполняется только в песочнице (отдельный процесс + таймаут, как `run_python` в `reference/baseline.py`), никогда через `exec` в процессе агента.
- Промпты — в файлах `agent/prompts/*.md`, не строками в коде. Каждый промпт содержит профиль данных и историю экспериментов, но не сырые данные.
- Бюджет: 600 ₽/день на всю команду. Отлаживай с `LLM_CACHE_DIR` и `LLM_RUB_LIMIT` из `.env`.

## Стыкуется с
ml (формат гипотез) · research (playbook, идеи гипотез) · captain (цикл в `solution.py`, бюджет) · qa (тесты устойчивости к ответам LLM).
