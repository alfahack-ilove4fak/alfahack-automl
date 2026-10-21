"""Разбор настоящих публичных задач и поиск метрики в тексте."""

from pathlib import Path

import pytest

from agent.task_parser import parse_task
from evaluation.metrics import compute_metric, find_metric_in_text
from interfaces.schemas import TaskType

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "folder, task_type, target, metric, n_classes",
    [
        ("regression", TaskType.REGRESSION, "site_eui", "rmse", None),
        ("multiclass", TaskType.MULTICLASS, "attack_cat", "f1_macro", 10),
    ],
)
def test_public_tasks_parsed(folder, task_type, target, metric, n_classes):
    spec = parse_task(ROOT / "tasks" / "public" / folder / "task.md")
    assert spec.task_type == task_type
    assert spec.target == target
    assert spec.id_column == "id"
    assert spec.metric.name == metric
    assert spec.train_path.exists() and spec.test_path.exists()
    assert (len(spec.class_labels) if spec.class_labels else None) == n_classes


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Метрика оценки — **RMSE** на исходной шкале", "rmse"),
        ("Основная метрика — **Macro F1**. Дополнительно Accuracy", "f1_macro"),
        ("Quality is measured by ROC-AUC.", "roc_auc"),
        ("Метрика: MAE", "mae"),
        ("Оцениваем по log loss", "logloss"),
        ("Без метрики", None),
        # Дополнительная метрика стоит раньше основной — побеждает та, что после «основная метрика»
        ("Дополнительно смотрим на Accuracy, но основная метрика — Macro F1", "f1_macro"),
        # «mse» не должна находиться внутри «rmse» и наоборот
        ("Метрика оценки: MSE (не путать с RMSE)", "mse"),
        # «weighted f1» не должна проиграть простому «f1-score»
        ("Submissions are scored using Weighted F1-score.", "f1_weighted"),
        # Падежи, полные названия и синонимы (Q-L1-3)
        ("Решения оцениваются по средней абсолютной ошибке", "mae"),
        ("Метрика — средняя абсолютная процентная ошибка (MAPE)", "mape"),
        ("Evaluation metric: Root Mean Squared Logarithmic Error", "rmsle"),
        ("Evaluation metric: Mean Squared Logarithmic Error", "msle"),
        ("Submissions are evaluated on area under the ROC curve", "roc_auc"),
        ("Метрика — macro-averaged F1", "f1_macro"),
        ("F1-score (macro)", "f1_macro"),
        ("Метрика — F1", "f1"),
        ("Основная метрика — F1 (гармоническое среднее precision и recall)", "f1"),
        ("Метрика — взвешенная F1", "f1_weighted"),
        ("Метрика: PR-AUC", "pr_auc"),
        ("Metric: AUC-PR", "pr_auc"),
        ("Submissions are scored by Average Precision", "pr_auc"),
        ("Метрика — коэффициент Джини (Gini = 2·AUC − 1)", "gini"),
        ("Evaluation: Normalized Gini coefficient", "gini"),
        ("Metric: Weighted AUC", "roc_auc_weighted"),
        ("Метрика — SMAPE", "smape"),
        ("Метрика: WAPE", "wape"),
        ("Качество оценивается коэффициентом детерминации", "r2"),
        ("Metric: R-squared", "r2"),
        ("Метрика: Precision", "precision"),
        ("Main metric is Recall of the positive class", "recall"),
        ("Оцениваем кросс-энтропией", "logloss"),
        ("Метрика: медианная абсолютная ошибка", "medae"),
        ("Метрика: квадратично взвешенная каппа", "qwk"),
        ("Метрика: сбалансированная точность", "balanced_accuracy"),
        ("Метрика: RMSE (корень из среднеквадратичной ошибки)", "rmse"),
        # Обычные слова в описании данных — не метрика (иначе агент решит, что это классификация)
        ("Полнота данных высокая, точность измерений 0.1 мм", None),
        # Отрицания (#85): упомянутая «не та» метрика пропускается
        (
            "**Evaluation.** Accuracy is misleading here, so we do NOT use it. Submissions are scored with\n"
            "binary cross-entropy on the predicted probability that the flow is an attack (target = 1).",
            "logloss",
        ),
        ("Метрика: accuracy здесь не подходит из-за дисбаланса, поэтому качество оценивается по BCE.", "logloss"),
        ("## Метрика\nВ отличие от accuracy, мы оцениваем вероятности: binary cross-entropy (log loss).", "logloss"),
        ("Метрика: Accuracy не используется. Используется logloss.", "logloss"),
        ("Submissions are evaluated using binary crossentropy between predicted probabilities and labels.", "logloss"),
        ("We use macro F1 instead of accuracy because classes are imbalanced.", "f1_macro"),
        ("Rather than RMSE, submissions are scored with MAE.", "mae"),
        ("Accuracy is reported for reference only; the evaluation metric is ROC-AUC.", "roc_auc"),
        ("Accuracy is not used. Use F1-score.", "f1"),
        # Отрицание относится к соседней метрике, а не ко всем подряд
        ("Основной метрикой выбран Macro F1, а не Accuracy", "f1_macro"),
        ("Метрика — F1 (accuracy не используется)", "f1"),
        ("Do not forget: the metric is RMSE.", "rmse"),
        ("Не забудьте: метрика — MAE.", "mae"),
    ],
)
def test_find_metric_in_text(text, expected):
    assert find_metric_in_text(text) == expected


def test_metric_directions():
    assert compute_metric("rmse", [1, 2], [1, 2]) == 0.0
    assert compute_metric("f1_macro", [0, 1], [0, 1]) == 1.0
