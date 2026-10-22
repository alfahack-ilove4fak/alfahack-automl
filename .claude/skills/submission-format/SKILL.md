---
name: submission-format
description: Правила формата сдачи на платформу Альфа Хак — интерфейс solution.py, ZIP, submission.csv, ограничения запуска. Используй при любых изменениях solution.py, записи ответа, упаковки или вопросах «примет ли платформа».
---

# Навык: формат сдачи

Источник истины — `reference/platform_instructions.md` (инструкция Boosters), PDF кейса и `reference/baseline.py`.

- **Архив**: в корне ровно `solution.py` + `requirements.txt`, **без папок** — платформа копирует только корневые `.py` и `requirements.txt`. Собирает `scripts/package_submission.py`: код агента упакован внутрь самораспаковывающегося `solution.py`. Никогда не собирай zip руками из папок репозитория — получишь 0 баллов. Проверка — `uv run pytest tests/smoke/test_package.py`.
- **Запуск**: `python3 solution.py task.md` в каталоге задачи; env `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`. Свой `pip` запускать нельзя; временные файлы — в каталоге задачи или `/tmp`.
- **Среда**: Docker, 8 CPU, 32 ГБ, Python 3.12, без интернета, только `requirements.txt`. `baseline.py` импортирует `mcp`, но в контейнере его нет (организаторы подтвердили) — не используем. На платформе нет лимита на вызовы LLM и токены, 600 ₽/день — только на разработку (`reference/organizer_chat_faq.md`).
- **Время**: 3 часа на задачу, затем принудительная остановка; оценивается файл, который есть к этому моменту.
- **Ответ**: `submission.csv` рядом с `task.md`, UTF-8, запятая, без индекса, колонки = заголовок `sample_submission.csv`, порядок `id` = `test.csv`, без пропусков/inf/дублей, метки только из допустимых классов. Проверка — `evaluation/submission.py::validate_submission`.
- **Пути**: пути в тексте `task.md` могут не совпадать с реальными — файлы ищет `agent/task_parser.py::find_task_files`.
- **Сабмиты**: 3 в день, сброс 00:00 МСК; на private берётся версия с лучшим public.
