# evaluation/ — роль: QA / MLOps (кто сейчас — docs/team.md, ревью — .github/CODEOWNERS)

Пакет роли: @docs/roles/qa.md · Материалы: docs/resources/qa.md

## Что здесь
| Файл | Роль |
|---|---|
| `metrics.py` | реестр метрик: функция, направление, что нужно на вход (value/label/proba), как называется в тексте |
| `submission.py` | `validate_submission` (все правила платформы) и `write_submission_atomic` |
| `local_score.py` | балл по формуле организаторов: 100·max(0, улучшение над бейзлайном); три бейзлайна — тривиальный ответ (`score`), сильный: лучший из LightGBM по умолчанию и тривиального (`model_score`), и сильнейший из набора «тривиальный, LightGBM, RandomForest 300, линейная» (`best_score`, `best_name`; оценка снизу, #276) |
| `harness.py` | «локальный лидерборд»: отложенная часть train как тест, прогон агента, баллы → `runs/harness/leaderboard.csv`; колонки `llm_*` — запросы, токены и ₽ Qwen из `run/llm_calls.jsonl` (тай-брейк); `shift_auc` — AUC «train против test». `--holdout test_like` — отложенная часть из строк train, самых похожих на test (при `shift_auc` ≥ 0.7), `--holdout both` — оба прогона и «чувствительность к сдвигу»; `--no-extra-baselines` — без RandomForest и линейной (быстрее). Запуск: `uv run python -m evaluation.harness --all --minutes 5`. В PR — метка `harness` → таблица «было/стало» (`.github/workflows/harness-pr.yml`, `scripts/harness_compare.py`) |

## Контракты
- Создаёшь: `ValidationReport`, реестр `METRICS`. Твой реестр используют agent (поиск метрики в тексте) и ml (CV).
- Добавляешь метрику → алиасы в `aliases`, место в `SEARCH_ORDER`, тест в `tests/unit/test_task_parsing.py`.

## Правила модуля
- Валидатор строже платформы: лучше ложная тревога локально, чем 0 баллов на платформе.
- Скрытые задачи могут прийти с любой метрикой — реестр должен покрывать всё разумное (RMSE, MAE, R², F1-варианты, ROC-AUC, LogLoss, MAPE, QWK…).
- Харнесс не должен «подсматривать»: отложенная часть train не участвует ни в обучении, ни в выборе гипотез.

## Стыкуется с
research (тренировочные задачи, локальный скор) · captain (проверка перед каждым сабмитом) · все (CI и doctor — тоже твоя зона: `scripts/`, `tests/`, `.github/workflows/`).
