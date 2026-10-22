# Проверенные материалы (пункт 3.6)

Правила раздела:
- Каждая ссылка **проверена** (открывается; у статей arXiv сверены номер и название) — дата проверки указана в файле.
- Рядом — **чем полезна именно нам**. Без этого ссылку не добавляем.
- Версии документации — те, что в `requirements.txt` (иначе API может отличаться).
- Нашёл хороший материал → PR в файл своей роли (Claude Code сам предложит). Не уверен в ссылке — помечай ⚠️.

| Роль | Файл |
|---|---|
| captain | [captain.md](captain.md) |
| agent | [agent.md](agent.md) |
| ml | [ml.md](ml.md) |
| research | [research.md](research.md) |
| qa | [qa.md](qa.md) |

## Общее для всех (проверено 2026-10-20)

| Материал | Чем полезен |
|---|---|
| [Pro Git на русском](https://git-scm.com/book/ru/v2) | если git совсем новый: главы 2–3 (основы и ветвление) — за час |
| [Learn Git Branching (рус.)](https://learngitbranching.js.org/?locale=ru_RU) | интерактивный тренажёр веток — 20 минут, и merge/rebase становятся понятны |
| [GitHub Docs: Pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests) | как устроены PR и ревью на GitHub |
| [GitHub Docs: Projects](https://docs.github.com/en/issues/planning-and-tracking-with-projects) | наша доска задач |
| [uv](https://docs.astral.sh/uv/) | менеджер Python и зависимостей проекта |
| [Claude Code — обзор](https://code.claude.com/docs/en/overview) | что умеет Claude Code |
| [Claude Code: best practices](https://www.anthropic.com/engineering/claude-code-best-practices) | как эффективно работать с Claude (план → код → проверка, контекст) |
| [Claude Code: сабагенты](https://code.claude.com/docs/en/sub-agents) и [скиллы](https://code.claude.com/docs/en/skills) | как устроены наши `.claude/agents` и `.claude/skills` |
| [Kaggle Learn: Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) | быстрый практический курс табличного ML: пропуски, категории, утечки, кросс-валидация |
| [mlcourse.ai](https://mlcourse.ai/) | открытый курс ML (автор — Юрий Кашницкий), есть материалы на русском |
| [Вебинар организаторов — разбор](webinar-qa.md) | ответы авторов кейса: что считается агентом, какие бывают скрытые задачи, ресурсы, тай-брейки, финал; расшифровка — `reference/webinar_transcript.md` |
| PDF кейса — `reference/case_alfahack_automl.pdf` | условия, ограничения, формулы баллов — первоисточник |
