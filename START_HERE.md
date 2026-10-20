# START_HERE — danil

👋 Привет! Ты в команде Альфа Хак. Это твоя личная инструкция — она лежит в твоей ветке `team/danil`.

| | |
|---|---|
| **Кто ты** | ML-pipeline инженер (`ml`), GitHub @marenkov17 |
| **Твоя зона** | `ml/` — данные, модели, кросс-валидация, финальное обучение |
| **Твой пакет роли** | [docs/roles/ml.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/docs/roles/ml.md) — что установить, инструменты, задачи L0–L3, Claude Code |
| **Проверенные материалы** | [docs/resources/ml.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/docs/resources/ml.md) |
| **Тебя ревьюит** | капитан @thenlaooo — пока в команде нет роли «Исследователь / Data Scientist» |

## 1. Что скачать и как проверить готовность (≈ 1 час)

Иди по [ONBOARDING.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/ONBOARDING.md) — 5 шагов. Сверх общей установки для твоей роли — раздел 3.1 пакета роли.
Финиш — команда:
```
uv run python scripts/doctor.py --role ml
```
→ должно быть `✅ ГОТОВ К РАБОТЕ`. При ❌ под пунктом написано, что выполнить. Застрял на 15+ минут — в чат команды.

**Правило:** к задачам кейса — только после зелёного doctor. Отметься в issue «CP0» на [доске](https://github.com/users/thenlaooo/projects/2).

## 2. Твоя первая задача

**M-L0-1 · Разогрев — прогнать эксперимент и прочитать ml/** — подробные шаги в пакете роли (раздел 3.4) и в issue на доске.
В Claude Code просто набери `/next-task`.

## 3. С кем стыкуешься

- **agent** — ежедневно в 13:00: какие `params` ты поддерживаешь.
- **research** — схема валидации, идеи признаков и моделей.
- **капитан** — финальное обучение и бюджет.

Ритм дня: стендап 10:00 · окна интеграции 13:00 и 18:30 · ретро 19:00 ([docs/team-process.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/docs/team-process.md)).

## 4. Как работаем с кодом

Ветка `ml/<задача>` от `dev` → PR в `dev` → ревью → слияние в окно интеграции. Шпаргалка: [docs/git-guide.md](https://github.com/thenlaooo/alfahack-automl/blob/dev/docs/git-guide.md).
⚠️ Эта ветка (`team/danil`) — только для тебя: заметки можно, PR из неё в `dev` — нельзя.

## 5. Мои заметки

_(пиши сюда что угодно: вопросы, идеи, что узнал)_
