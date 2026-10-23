"""Ночной аудит: как find_metric_in_text понимает разные формулировки метрики (без LLM, секунды).

Запуск: uv run python research/night/metric_phrases.py
Для каждой фразы — ожидаемая метрика реестра (None — «в тексте нет метрики»). Печатает расхождения.
"""

from evaluation.metrics import METRICS, find_metric_in_text

CASES = [
    # (текст, ожидаемый ключ реестра)
    ("Metric: MAPE. Lower is better.", "mape"),
    ("Оценка: средняя абсолютная процентная ошибка (MAPE), чем меньше, тем лучше.", "mape"),
    ("Качество измеряется симметричной MAPE (SMAPE).", "smape"),
    ("Solutions are evaluated by the coefficient of determination R^2.", "r2"),
    ("Метрика — коэффициент детерминации R2.", "r2"),
    ("Evaluation: Root Mean Squared Logarithmic Error (RMSLE).", "rmsle"),
    ("Метрика: среднеквадратичная логарифмическая ошибка.", "rmsle"),
    ("Оценивается по MAE (средняя абсолютная ошибка).", "mae"),
    ("Score = mean absolute error", "mae"),
    ("Метрика качества: log loss (логарифмическая функция потерь).", "logloss"),
    ("Metric: binary cross-entropy.", "logloss"),
    ("Оценка — площадь под ROC-кривой (AUC).", "roc_auc"),
    ("The metric is area under the precision-recall curve (PR AUC).", "pr_auc"),
    ("Метрика — AUC-PR.", "pr_auc"),
    ("Average precision is the target metric.", "pr_auc"),
    ("Метрика: F-beta, beta=2.", "f1"),  # F-beta нет в реестре — ближайшая F1; фиксируем, что ищется хоть что-то
    ("Quality: F2 score", "f1"),
    ("Cohen's kappa (quadratic weighted) is used.", "qwk"),
    ("Метрика — квадратичный взвешенный каппа Коэна.", "qwk"),
    ("Metric: balanced accuracy.", "balanced_accuracy"),
    ("Метрика — сбалансированная точность.", "balanced_accuracy"),
    ("Evaluation: MAP@5.", "map"),
    ("Метрика: MAP@10 (mean average precision at k).", "map"),
    ("NDCG@10 is used for ranking.", "ndcg"),
    ("Quantile loss with alpha 0.9.", "pinball@0.9"),
    ("Метрика: квантильная ошибка, квантиль 0,75.", "pinball@0.75"),
    ("The Matthews correlation coefficient decides the winner.", "mcc"),
    ("Macro-averaged F1.", "f1_macro"),
    ("Метрика — макро F1-мера.", "f1_macro"),
    ("Weighted F1 score.", "f1_weighted"),
    ("Мера качества — доля верных ответов.", "accuracy"),
    ("Мера качества — точность (accuracy).", "accuracy"),
    ("Среднеквадратичная ошибка (RMSE).", "rmse"),
    ("Mean squared error is used.", "mse"),
    ("Lower values of the score are better. Metric: Gini coefficient.", "gini"),
    ("Оценка: нормализованный коэффициент Джини.", "gini"),
    ("Решения оцениваются по метрике WAPE.", "wape"),
    ("Score: median absolute error", "medae"),
    ("We evaluate with Tweedie deviance.", "tweedie_deviance"),
    ("Решения сравниваются на скрытой выборке по специальной метрике качества организаторов.", None),
    ("Predict the price. We also use accuracy only for sanity, the main score is RMSE.", "rmse"),
    ("Не используйте accuracy; оценка — ROC AUC.", "roc_auc"),
    ("# Задача\nПредскажите отток.\n\nОценка: Recall@k по топ-10% клиентов.", None),  # нет в реестре — честный None лучше, чем recall
    ("Evaluation metric: Brier score.", None),
    ("Метрика: ошибка в процентах MAPE, считается только по строкам, где факт > 0.", "mape"),
    ("Metric: logarithmic loss", "logloss"),
    ("Score: exponentiated RMSE (RMSE on log1p target)", "rmsle"),
]

bad = 0
for text, expected in CASES:
    got = find_metric_in_text(text)
    ok = got == expected
    bad += not ok
    print(("OK  " if ok else "DIFF"), f"{expected!s:18} <- {got!s:18} | {text[:70]!r}")
print(f"\nрасхождений: {bad} из {len(CASES)}; метрик в реестре: {len(METRICS)}")
