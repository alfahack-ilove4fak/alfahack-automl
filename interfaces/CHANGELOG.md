# Журнал изменений контрактов

Правило: любое изменение `interfaces/` → новая строка здесь + повышение `CONTRACT_VERSION` в `schemas.py`.
- **patch** (1.0.x) — добавили необязательное поле со значением по умолчанию;
- **minor** (1.x.0) — добавили новый контракт или функцию;
- **major** (x.0.0) — переименовали/удалили поле, поменяли смысл. Только на синке всей команды.

| Версия | Дата | Кто | Что изменилось | Кого затрагивает |
|---|---|---|---|---|
| 1.0.0 | 2026-10-20 | @thenlaooo | Первая версия: TaskSpec, DataProfile, Hypothesis, ExperimentResult, RunBudget, ValidationReport | все |
| 1.1.0 | 2026-10-21 | @thenlaooo | ADR 0008: `ExperimentResult.validation` (схема CV, `beats()` не сравнивает разные схемы) и `refit_params` (вместо скрытой правки `hypothesis.params`); `budget` в `ProposeHypotheses`; новый `CodeResult` и протокол `RunCode`; задокументированы `Hypothesis.code`, ключи `DataProfile.target_summary`, формат `oof_path`/`test_pred_path` | ml, agent, captain |
| 1.1.1 | 2026-10-22 | @n0h4ndy (по добру @thenlaooo, #174) | `TaskSpec.extra_targets: list[str] = []` — остальные таргеты задачи с несколькими колонками ответа; `target` по-прежнему основной | agent (заполняет), captain (`solution.py`: ответ по всем таргетам) |
| 1.1.2 | 2026-10-23 | @thenlaooo | `ROW_ID_COLUMN` — значение `TaskSpec.id_column` для задач без id: ответ по порядку строк test (в шаблоне нет id-колонки). Поля и типы не менялись: `ml.profiling.load_data` добавляет колонку-номер в test, а в ответ она не входит (`submission_columns` без неё). Если в шаблоне id есть, а в test нет — значения id берутся из sample_submission | agent (`parse_task` ставит), ml (`load_data` добавляет), evaluation (`validate_submission` не требует id в test, если его нет в ответе) |
