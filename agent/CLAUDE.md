# agent/ — роль: LLM/agent-инженер (кто сейчас — docs/team.md, ревью — .github/CODEOWNERS)

Пакет роли: @docs/roles/agent.md · Материалы: docs/resources/agent.md

## Что здесь
| Файл | Роль |
|---|---|
| `task_parser.py` | `task.md` + файлы → `TaskSpec`. Детерминированный, работает без LLM — не ломать! |
| `llm_client.py` | Qwen через OpenAI-совместимый API: кэш на диск, учёт ₽, `available()` = False без ключа; по остатку бюджета (`seconds_left`) ждёт не дольше четверти оставшегося времени, при < 4 мин не зовёт LLM (ответ Qwen ~35 с) |
| `hypotheses.py` | `propose_hypotheses(spec, profile, history, k, budget)`: сначала LLM (если доступна и хватает времени), недостающие — из `research/playbook.yaml`; повторы id/params отсекаются |
| `llm_hypotheses.py` | ответ LLM → проверенные `Hypothesis`: схема, допустимые kind и ключи params, без повторов |
| `prompting.py` | сборка промптов из `agent/prompts/*.md` (string.Template, `$плейсхолдеры`); нет файла → `None` → playbook |
| `feature_code.py` | код признаков от LLM: промпт `prompts/features.md`, разбор ответа, проверка кода до песочницы (AST); `apply_feature_code(code, df)` — запуск в песочнице, возвращает df + новые числовые колонки или причину |
| `loop.py` | цикл AutoResearch: гипотеза → `ml.run_experiment` → принять, если лучше на `min_delta` |
| `memory.py` | память цикла: решение по каждой гипотезе и последние выводы LLM после неудачных пачек |
| `critic.py` | критик по правилам: пустые для задачи гипотезы, недопустимые значения, цена по времени → пропуск (`SKIPPED`) до запуска |
| `llm_critic.py` | LLM-критик: одна проверка пачки LLM-гипотез до запуска (утечка, подгонка, метрика, повтор, цена); `AGENT_LLM_CRITIC=1`, при сбое пропускает всё |
| `loop.py` | цикл AutoResearch: очередь гипотез (по 3 за запрос к LLM) → `ml.run_experiment` → принять, если лучше на `min_delta` |

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
