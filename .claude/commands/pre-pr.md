---
description: Проверки перед Pull Request — линтер, тесты, контракты, зона ответственности, секреты
allowed-tools: Bash(uv run ruff *), Bash(uv run pytest *), Bash(git status *), Bash(git diff *), Bash(git fetch *), Bash(git log *), Bash(git branch *), Grep, Read
---

Проведи участника через проверки перед PR. Выведи чек-лист с ✅/❌ и что исправить.

1. **Ветка**: текущая не `main`/`dev`/`team/*`; имя вида `<роль>/<задача>`. `git fetch origin` и проверь, нет ли отставания от `origin/dev` (если есть — предложи `git merge origin/dev`).
2. **Линтер**: `uv run ruff check .`
3. **Тесты**: `uv run pytest -q` (дымовые идут 1–2 минуты — предупреди).
4. **Зона**: `git diff origin/dev...HEAD --name-only` → все ли файлы в зоне роли участника по `.github/CODEOWNERS`. Чужие файлы — согласованы ли с владельцем?
5. **Контракты**: изменён `interfaces/` → нужен `/sync-check` и отдельный PR по процессу.
6. **Секреты и мусор**: нет `.env`, ключей (поиск по `sk-`, `api_key=`, `OPENAI_API_KEY=` с значением), больших файлов, `submission.csv`, `runs/`, ноутбуков с выводом на мегабайты.
7. **Тест на новое**: есть ли тест на новое поведение? Если нет — предложи сабагента `test-writer`.
8. **Журнал**: если это эксперимент или меняет качество — строка в `docs/experiments.md`.

Если всё ✅ — подготовь заголовок PR (`тип: что сделано`), заполненное описание по `.github/pull_request_template.md` и команду
`git push -u origin <ветка> && gh pr create --base dev --title "..." --body-file <файл> --reviewer <ник ревьюера из CONTRIBUTING.md>`.
Сам push/PR выполняй только после подтверждения участника.
