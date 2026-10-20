---
name: submission-format
description: Правила формата сдачи на платформу Альфа Хак — интерфейс solution.py, ZIP, submission.csv, ограничения запуска. Используй при любых изменениях solution.py, записи ответа, упаковки или вопросах «примет ли платформа».
---

# Навык: формат сдачи

Источник истины — PDF кейса (`reference/case_alfahack_automl.pdf`) и `reference/baseline.py`.

- **Архив**: ZIP, в корне `solution.py`; остальные модули рядом. Собирает `scripts/package_submission.py`.
- **Запуск**: `python solution.py <путь>/task.md`; env `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`. ⚠️ В приложении 2 упомянут `script.py` — уточнить у организаторов.
- **Среда**: Docker, 8 CPU, 32 ГБ, Python 3.12, без интернета, только `requirements.txt`. ⚠️ `baseline.py` импортирует `mcp`, которого нет в `requirements.txt` — не используем.
- **Время**: 3 часа на задачу, затем принудительная остановка; оценивается файл, который есть к этому моменту.
- **Ответ**: `submission.csv` рядом с `task.md`, UTF-8, запятая, без индекса, колонки = заголовок `sample_submission.csv`, порядок `id` = `test.csv`, без пропусков/inf/дублей, метки только из допустимых классов. Проверка — `evaluation/submission.py::validate_submission`.
- **Пути**: пути в тексте `task.md` могут не совпадать с реальными — файлы ищет `agent/task_parser.py::find_task_files`.
- **Сабмиты**: 3 в день, сброс 00:00 МСК; на private берётся версия с лучшим public.
