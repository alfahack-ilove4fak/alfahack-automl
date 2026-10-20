# START_HERE — egor

👋 Привет! Ты в команде Альфа Хак. Это твоя личная инструкция — она лежит в твоей ветке `team/egor`.

| | |
|---|---|
| **Кто ты** | QA / MLOps (`qa`), GitHub @dzigoku-ku |
| **Твоя зона** | `evaluation/`, `scripts/`, `tests/`, `.github/workflows/` — качество, харнесс, CI, сабмиты |
| **Твой пакет роли** | [docs/roles/qa.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/docs/roles/qa.md) — что установить, инструменты, задачи L0–L3, Claude Code |
| **Проверенные материалы** | [docs/resources/qa.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/docs/resources/qa.md) |
| **Тебя ревьюит** | @thenlaooo (Капитан / интегратор) |

## 1. Что скачать и как проверить готовность (≈ 1 час)

Иди по [ONBOARDING.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/ONBOARDING.md) — 5 шагов. Сверх общей установки для твоей роли — раздел 3.1 пакета роли.
Финиш — команда:
```
uv run python scripts/doctor.py --role qa
```
→ должно быть `✅ ГОТОВ К РАБОТЕ`. При ❌ под пунктом написано, что выполнить. Застрял на 15+ минут — в чат команды.

**Правило:** к задачам кейса — только после зелёного doctor. Отметься в issue «CP0» на [доске](https://github.com/users/thenlaooo/projects/2).

## 2. Твоя первая задача

**Q-L0-1 · Разогрев — тесты, CI, doctor** — подробные шаги в пакете роли (раздел 3.4) и в issue на доске.
В Claude Code просто набери `/next-task`.

## 3. С кем стыкуешься

- **research** — 22.10 18:30: тренировочные задачи для харнесса.
- **капитан** — проверка перед каждым сабмитом.
- **все** — помогаешь пройти CP0, разбираешь красный CI.

Ритм дня: стендап 10:00 · окна интеграции 13:00 и 18:30 · ретро 19:00 ([docs/team-process.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/docs/team-process.md)).

## 4. Как работаем с кодом

Ветка `qa/<задача>` от `dev` → PR в `dev` → ревью → слияние в окно интеграции. Шпаргалка: [docs/git-guide.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/docs/git-guide.md).
⚠️ Эта ветка (`team/egor`) — только для тебя: заметки можно, PR из неё в `dev` — нельзя.

## 5. Мои заметки

_(пиши сюда что угодно: вопросы, идеи, что узнал)_
