# ml/ — роль: ML-pipeline инженер (кто сейчас — docs/team.md, ревью — .github/CODEOWNERS)

Пакет роли: @docs/roles/ml.md · Материалы: docs/resources/ml.md

## Что здесь
| Файл | Роль |
|---|---|
| `profiling.py` | загрузка CSV, `DataProfile`, матрица признаков (`make_matrix`) |
| `experiment.py` | `run_experiment` (CV по гипотезе → `ExperimentResult`) и `fit_predict_final` (весь train → ответ) |

## Контракты
- Создаёшь: `DataProfile`, `ExperimentResult`, предсказания для сабмита. Читаешь: `TaskSpec`, `Hypothesis`, `RunBudget`.
- Поддерживаемые ключи `Hypothesis.params` перечислены в docstring `experiment.py`. Добавил ключ → обнови docstring, `research/playbook.yaml` (пример) и скажи agent-инженеру.

## Правила модуля
- **Никаких утечек**: всё, что «учится» (кодировщики, нормировки, отбор признаков), обучается только на train-части фолда.
- Метрика — только через `evaluation.metrics.compute_metric` и `METRICS[name].prediction` (value/label/proba).
- Уважай `RunBudget`: проверяй `budget.has_time()` между фолдами; оценивай память до обучения тяжёлых моделей.
- Ошибка гипотезы → `ExperimentResult(status=FAILED, error=...)`, а не исключение.
- `random_state` фиксирован (SEED) — результаты должны воспроизводиться.
- Тяжёлые пакеты (catboost, xgboost, autogluon) — из группы `full`: `uv sync --group full`. Тест на них в CI — через задачу QA.

## Стыкуется с
agent (формат гипотез, 13:00) · research (идеи признаков и моделей, валидация) · qa (метрики, харнесс) · captain (`fit_predict_final` в `solution.py`).
