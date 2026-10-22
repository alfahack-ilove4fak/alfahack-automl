---
description: Проверки перед сабмитом на платформу (капитан и QA)
allowed-tools: Bash(git *), Bash(gh run list *), Bash(gh run view *), Bash(uv run *), Read
---

Сабмитов всего ~3 в день — каждый должен быть осмысленным. Пройди чек-лист и выведи ✅/❌:

1. Сабмитим из **тега на `main`**: `main` содержит нужный PR из `dev`, CI на `main` зелёный (`gh run list --branch main --limit 3`).
2. Сколько сабмитов сегодня уже было — по `docs/submissions.md` (лимит 3, сброс в 00:00 МСК).
3. Что нового по сравнению с прошлым сабмитом и почему ждём улучшения (локальный скор/харнесс: было → стало).
4. Нет ли в коде «подгонки» под публичные задачи (имена колонок `site_eui`, `attack_cat`, пути `tasks/public` в коде агента).
5. Сборка: `uv run python scripts/package_submission.py` → zip, в корне **только** `solution.py` и `requirements.txt` (скрипт сам это проверяет). Тест «как на платформе»: `uv run pytest tests/smoke/test_package.py`.
6. Прогон из архива на обеих публичных задачах в чистой папке с бюджетом — `uv run python scripts/pre_submit.py`, итог «✅ можно сабмитить». Скрипт сам проверяет и при нарушении даёт ❌:
   - архив ≤ 100 МБ, в корне только `solution.py` и `requirements.txt`;
   - `requirements.txt` == репозиторий == `reference/requirements.txt` (набор платформы);
   - в коде агента нет `os.rename` / `shutil.move` (перенос между файловыми системами давал 0 баллов, #148), нет `pip`, импорты только из `requirements.txt`;
   - внутри нет чужих библиотек (`.whl`, `.so`, `site-packages`) — только наш код;
   - запуск из пустой папки без кэшей прошлых запусков (платформа их не хранит), в логе есть «Сохранён ответ «baseline»», `submission.csv` валиден.
   Руками проверь только то, что скрипт не умеет: seed зафиксированы (`SEED` в `ml/`, `random_state` в новых моделях).
7. Команды тега: `git tag sub-ГГГГ-ММ-ДД-N origin/main && git push origin sub-ГГГГ-ММ-ДД-N` (push — после подтверждения).
8. Напомни: после загрузки — строка в `docs/submissions.md` (локально vs public).
