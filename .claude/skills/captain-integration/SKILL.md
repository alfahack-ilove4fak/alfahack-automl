---
name: captain-integration
description: Работа капитана/интегратора — solution.py, runtime (бюджет, логи, песочница), слияние dev→main, релизы и теги сабмитов, подключение участников, доска задач, контроль чек-поинтов. Используй, когда задача касается solution.py, runtime/, релиза, add_member.py или процесса команды.
---

# Навык: интеграция и релизы

Пакет роли: `docs/roles/captain.md` · материалы: `docs/resources/captain.md` · модуль: `runtime/CLAUDE.md`.

## Рецепт: окно интеграции (13:00 и 18:30)
1. `gh pr list --base dev` → готовые PR (approve + зелёный CI). Сначала PR с `interfaces/`, потом остальные.
2. После слияний: `git switch dev && git pull && uv run pytest && uv run python scripts/run_local.py --all --minutes 5 --mock-llm`.
3. Красное → откат через `git revert` в PR или быстрый фикс владельцем.

## Рецепт: релиз и сабмит
1. PR `dev → main` («релиз: …»), ревью QA, слияние только тобой.
2. `/submit-check` → тег `sub-ГГГГ-ММ-ДД-N` → загрузка → строка в `docs/submissions.md`.

## Рецепт: новый участник
`python3 scripts/add_member.py --name <латиница> --github <ник> --role <роль>` → слить созданный PR → доступ к доске → сообщение в чат (скрипт печатает).

## Принципы
- `solution.py` всегда отдаёт валидный ответ: тривиальный → бейзлайн → улучшения.
- Чек-поинты (`docs/checkpoints.md`) — не формальность: на стендапе смотри, кто блокирует ближайший.
