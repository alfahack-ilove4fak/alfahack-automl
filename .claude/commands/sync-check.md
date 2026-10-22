---
description: Не сломал ли я контракт с другими модулями? Проверка перед изменением interfaces/ и перед PR
allowed-tools: Bash(git diff *), Bash(git fetch *), Bash(git log *), Bash(uv run pytest *), Grep, Glob, Read
---

Проверь, совместимы ли изменения участника с контрактами команды.

1. `git fetch origin && git diff origin/dev...HEAD --stat` и полный diff по `interfaces/`, а также по публичным функциям из `interfaces/protocols.py` (parse_task, propose_hypotheses, profile_data, run_experiment, fit_predict_final, compute_metric, validate_submission).
2. Для каждого изменённого поля/функции найди всех потребителей (Grep по репозиторию) и их владельцев (`.github/CODEOWNERS`).
3. Проверь процесс (если менялся `interfaces/`): повышен ли `CONTRACT_VERSION` и правильного ли уровня (patch/minor/major), есть ли строка в `interfaces/CHANGELOG.md`, обновлены ли `stubs.py` и `tests/contracts`.
4. Новые ключи в `Hypothesis.params` — есть ли они в docstring `ml/experiment.py` и согласованы ли с владельцем ml.
5. Запусти `uv run pytest tests/contracts -q`.

Итог — таблица: «что изменилось → кого затрагивает (@ник) → совместимо? → что сделать». В конце — готовый текст сообщения в чат владельцам, если нужно согласование. Ничего не исправляй сам без просьбы.
