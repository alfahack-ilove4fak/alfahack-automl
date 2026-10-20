# START_HERE — vasya

👋 Привет! Ты в команде Альфа Хак. Это твоя личная инструкция — она лежит в твоей ветке `team/vasya`.

| | |
|---|---|
| **Кто ты** | Капитан / интегратор (`captain`), GitHub @thenlaooo |
| **Твоя зона** | `solution.py`, `runtime/`, `interfaces/` (по процессу), релизы и сабмиты |
| **Твой пакет роли** | [docs/roles/captain.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/docs/roles/captain.md) — что установить, инструменты, задачи L0–L3, Claude Code |
| **Проверенные материалы** | [docs/resources/captain.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/docs/resources/captain.md) |
| **Тебя ревьюит** | @dzigoku-ku (QA / MLOps) |

## 1. Что скачать и как проверить готовность (≈ 1 час)

Иди по [ONBOARDING.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/ONBOARDING.md) — 5 шагов. Сверх общей установки для твоей роли — раздел 3.1 пакета роли.
Финиш — команда:
```
uv run python scripts/doctor.py --role captain
```
→ должно быть `✅ ГОТОВ К РАБОТЕ`. При ❌ под пунктом написано, что выполнить. Застрял на 15+ минут — в чат команды.

**Правило:** к задачам кейса — только после зелёного doctor. Отметься в issue «CP0» на [доске](https://github.com/orgs/alfahack-ilove4fak/projects/1).

## 2. Твоя первая задача

**C-L0-1 · Провести CP0 — у всех зелёный doctor** — подробные шаги в пакете роли (раздел 3.4) и в issue на доске.
В Claude Code просто набери `/next-task`.

## 3. С кем стыкуешься

- **qa** — проверка перед каждым сабмитом и релизы `dev → main`.
- **agent** — цикл AutoResearch и бюджет времени.
- **все** — стендап, окна интеграции, контракты.

Ритм дня: стендап 10:00 · окна интеграции 13:00 и 18:30 · ретро 19:00 ([docs/team-process.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/docs/team-process.md)).

## 4. Как работаем с кодом

Ветка `captain/<задача>` от `dev` → PR в `dev` → ревью → слияние в окно интеграции. Шпаргалка: [docs/git-guide.md](https://github.com/alfahack-ilove4fak/alfahack-automl/blob/dev/docs/git-guide.md).
⚠️ Эта ветка (`team/vasya`) — только для тебя: заметки можно, PR из неё в `dev` — нельзя.

## 5. Мои заметки

_(пиши сюда что угодно: вопросы, идеи, что узнал)_
