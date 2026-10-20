# Ролевые пакеты (пункт 3)

| Роль | Пакет | Материалы | Кто сейчас |
|---|---|---|---|
| Капитан / интегратор | [captain.md](captain.md) | [resources/captain.md](../resources/captain.md) | @thenlaooo |
| LLM/agent-инженер | [agent.md](agent.md) | [resources/agent.md](../resources/agent.md) | @n0h4ndy |
| ML-pipeline инженер | [ml.md](ml.md) | [resources/ml.md](../resources/ml.md) | — |
| Исследователь / DS | [research.md](research.md) | [resources/research.md](../resources/research.md) | — |
| QA / MLOps | [qa.md](qa.md) | [resources/qa.md](../resources/qa.md) | @dzigoku-ku |

Актуальный состав — [docs/team.md](../team.md).

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
Роли без человека (ml, research): их L0–L1 в первые дни берут капитан и свободные участники по договорённости на стендапе.
