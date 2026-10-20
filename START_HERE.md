# START_HERE — artem

👋 Привет! Ты в команде Альфа Хак. Это твоя личная инструкция — она лежит в твоей ветке `team/artem`.

| | |
|---|---|
| **Кто ты** | LLM/agent-инженер (`agent`), GitHub @n0h4ndy |
| **Твоя зона** | `agent/` — разбор задачи, LLM, гипотезы, цикл AutoResearch |
| **Твой пакет роли** | [docs/roles/agent.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/docs/roles/agent.md) — что установить, инструменты, задачи L0–L3, Claude Code |
| **Проверенные материалы** | [docs/resources/agent.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/docs/resources/agent.md) |
| **Тебя ревьюит** | @marenkov17 (ML-pipeline инженер) |

## 1. Что скачать и как проверить готовность (≈ 1 час)

Иди по [ONBOARDING.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/ONBOARDING.md) — 5 шагов. Сверх общей установки для твоей роли — раздел 3.1 пакета роли.
Финиш — команда:
```
uv run python scripts/doctor.py --role agent
```
→ должно быть `✅ ГОТОВ К РАБОТЕ`. При ❌ под пунктом написано, что выполнить. Застрял на 15+ минут — в чат команды.

**Правило:** к задачам кейса — только после зелёного doctor. Отметься в issue «CP0» на [доске](https://github.com/orgs/alfahack-ilove4fak/projects/1).

## 2. Твоя первая задача

**A-L0-1 · Разогрев — запустить агента и разобраться в цикле** — подробные шаги в пакете роли (раздел 3.4) и в issue на доске.
В Claude Code просто набери `/next-task`.

## 3. С кем стыкуешься

- **ml** — ежедневно в 13:00: формат гипотез (`Hypothesis.params`).
- **research** — 22.10 13:00: playbook v1 (примеры гипотез).
- **капитан** — ключ LLM (лично), песочница для кода LLM, доли бюджета.

Ритм дня: стендап 10:00 · окна интеграции 13:00 и 18:30 · ретро 19:00 ([docs/team-process.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/docs/team-process.md)).

## 4. Как работаем с кодом

Ветка `agent/<задача>` от `dev` → PR в `dev` → ревью → слияние в окно интеграции. Шпаргалка: [docs/git-guide.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/docs/git-guide.md).
⚠️ Эта ветка (`team/artem`) — только для тебя: заметки можно, PR из неё в `dev` — нельзя.

## 5. Мои заметки

_(пиши сюда что угодно: вопросы, идеи, что узнал)_
