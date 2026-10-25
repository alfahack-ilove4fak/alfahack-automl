# ml/ — роль: ML-pipeline инженер (кто сейчас — docs/team.md, ревью — .github/CODEOWNERS)

Пакет роли: @docs/roles/ml.md · Материалы: docs/resources/ml.md

## Что здесь
| Файл | Роль |
|---|---|
| `profiling.py` | загрузка CSV, `DataProfile`, матрица признаков (`make_matrix`) |
| `shift.py` | сдвиг train→test (#294): AUC «строка из train или из test» и признаки сдвига → `agent.log` и `DataProfile.warnings` строкой `adversarial_auc=0.88; …` (AUC без колонки времени/групп, выбранной `choose_validation`); только измерение, решений не меняет; `AGENT_SHIFT=0` выключает |
| `experiment.py` | `run_experiment` (CV по гипотезе → `ExperimentResult`) и `fit_predict_final` (весь train → ответ; для одиночных LightGBM/XGBoost/CatBoost — среднее нескольких сидов, `AGENT_FINAL_SEEDS`: по умолчанию 5 до 10 млн ячеек, 3 до 25 млн, больше один, `1` — выключено, #261). Модели: lightgbm, catboost, xgboost, hist_gb, linear, random_forest, extra_trees |

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
