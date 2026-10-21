# Журнал экспериментов

Сюда — **каждый** эксперимент, даже неудачный: неудачи экономят время другим и идут в питч.
Скор — по локальной кросс-валидации (или харнессу), не по платформе. Платформенные результаты — в [submissions.md](submissions.md).

Как записать быстро: в Claude Code «запиши эксперимент в журнал» (скилл `experiment-log`).

| Дата | Кто | Задача | Гипотеза | Было | Стало | Вывод | Ветка/PR |
|---|---|---|---|---|---|---|---|
| 2026-10-20 | @thenlaooo | multiclass | Скелет без LLM: бейзлайн LightGBM + playbook (5 гипотез), 15 мин | — | CV Macro F1 0.567 → **0.604** (lgbm_slow_lr, lgbm_small_leaves) | playbook помогает; balanced_classes и big_leaves хуже | run-agent #1, PR #2 |
| 2026-10-20 | @thenlaooo | regression | Скелет без LLM: бейзлайн LightGBM + playbook, 15 мин | — | CV RMSE **41.81** (бейзлайн) | lgbm_slow_lr дал 41.62, но отсеян порогом шума 0.32 (A-L2-3); вероятная утечка в CV — здания по годам (R-L2-1) | run-agent #1, PR #2 |
| 2026-10-21 | @Richi-Marqez | regression | Схема валидации: KFold vs GroupKFold(здание) vs время (годы 1–4 → 5); быстрый LightGBM | KFold RMSE 41.95 ± 1.17 | GroupKFold 43.89 ± 1.38; время 46.92 | KFold оптимистичен на ~5 RMSE: test = год 6, 35% строк — повторы зданий. Нужна валидация по времени (R-L2-1) | research/r-l1-1-eda, docs/research/eda.md |
| 2026-10-21 | @Richi-Marqez | multiclass | Схема валидации: KFold vs StratifiedKFold vs StratifiedGroupKFold(дубликаты); быстрый LightGBM | StratifiedKFold Macro F1 0.5678 ± 0.0065 | KFold 0.5509 ± 0.0110; SGKF 0.5497 ± 0.0080 | стратификация обязательна; 45% train — дубликаты (17% с разными метками) → CV завышена на 0.01–0.02 | research/r-l1-1-eda, docs/research/eda.md |
| 2026-10-21 | @Richi-Marqez | обе | Playbook v1: 15 кандидатов через стенд research/bench_playbook.py — cv агента и честная отложенная часть (время / группы дубликатов) | бейзлайн: F1 cv 0.5675 / honest 0.5628; RMSE cv 41.81 / honest 48.40 | лучшие на honest: balanced_classes F1 0.618; extra_trees RMSE 42.51, smooth_leaves 44.52 | 11 гипотез оставлены, 4 выброшены; cv и honest часто расходятся по знаку — нужна честная схема валидации (ADR 0008 D1) | research/playbook-v1, docs/research/playbook-v1.md |
| 2026-10-21 | @n0h4ndy | regression + multiclass | Правило принятия: «нулевые» гипотезы (другой seed) и playbook против бейзлайна, попарно по фолдам | `0.25·std`: 4 ошибки из 8 (3 шумовые приняты на multiclass, lgbm_slow_lr отклонена на regression) | t Nadeau–Bengio ≥ 2: 0 ошибок; шум t ≤ 1.73, улучшения 2.04–2.38 | шум от seed на Macro F1 (до 0.016) больше 0.25·std; порог 2.0 с небольшим запасом | ADR 0010, A-L2-3 |
