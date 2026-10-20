# Ролевые пакеты (пункт 3)

<!-- team:start (генерирует scripts/team_sync.py — не править руками) -->

| Роль | Пакет | Материалы | Кто сейчас |
|---|---|---|---|
| Капитан / интегратор | [captain.md](captain.md) | [resources/captain.md](../resources/captain.md) | @thenlaooo |
| LLM/agent-инженер | [agent.md](agent.md) | [resources/agent.md](../resources/agent.md) | @n0h4ndy ⏳ |
| ML-pipeline инженер | [ml.md](ml.md) | [resources/ml.md](../resources/ml.md) | @marenkov17 |
| Исследователь / Data Scientist | [research.md](research.md) | [resources/research.md](../resources/research.md) | — (свободно) |
| QA / MLOps | [qa.md](qa.md) | [resources/qa.md](../resources/qa.md) | @dzigoku-ku |

⏳ — приглашение в организацию ещё не принято. Свободные роли: Исследователь / Data Scientist — их задачи L0–L1 раздаём на стендапе.

<!-- team:end -->

Полный состав с личными ветками — [docs/team.md](../team.md). Обновить статусы: `uv run python scripts/team_sync.py`.

## Что в каждом пакете

| Раздел | Что внутри |
|---|---|
| 3.1 Установка | что поставить сверх общей [docs/install.md](../install.md), команды по ОС, как проверить |
| 3.2 Git в роли | особенности твоей роли поверх общего [docs/git-guide.md](../git-guide.md) |
| 3.3 Инструменты | библиотеки, сервисы, данные, LLM — и зачем каждый именно тебе |
| 3.4 Чек-лист L0–L3 | задачи шаг за шагом, критерий «готово», стыковки 🔗 |
| 3.5 Claude Code | твои скиллы, команды, сабагенты, примеры запросов |

## Откуда задачи

Единый источник — [tasks.yaml](tasks.yaml). Чек-листы в пакетах и issue на доске генерирует `scripts/tasks.py`:
```
uv run python scripts/tasks.py render     # обновить чек-листы в пакетах
uv run python scripts/tasks.py issues     # завести недостающие задачи на доске
```
Уровни: **L0** — разогрев (первый час), **L1** — базовые, **L2** — основные, **L3** — «на победу».
Если у роли нет человека — её задачи L0–L1 берут капитан и свободные участники по договорённости на стендапе.
