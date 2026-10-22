"""Разбор настоящих публичных задач и поиск метрики в тексте."""

from pathlib import Path

import pandas as pd
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


def _task(tmp_path, train, test, text="Метрика — RMSE.", sample=None, dictionary=None) -> Path:
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    if sample is not None:
        sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    if dictionary is not None:
        dictionary.to_csv(tmp_path / "data_dictionary.csv", index=False)
    (tmp_path / "task.md").write_text(text, encoding="utf-8")
    return tmp_path / "task.md"


def test_dictionary_columns_case_insensitive(tmp_path):
    # формат словаря не гарантирован (вебинар): «Feature, Role» вместо «feature, role»
    train = pd.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [0.1, 0.2, 0.3], "info": ["a", "b", "c"]})
    test = pd.DataFrame({"id": [4, 5], "x": [4.0, 5.0]})
    dictionary = pd.DataFrame({"Feature": ["id", "x", "y", "info"], "Role": ["ID", "feature", "Target", "info"]})
    spec = parse_task(_task(tmp_path, train, test, dictionary=dictionary))
    assert spec.target == "y" and spec.id_column == "id"


def test_empty_target_column_in_test(tmp_path):
    # target есть и в test, но пустой — заготовка под ответ; sample_submission нет
    train = pd.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [0.1, 0.2, 0.3]})
    test = pd.DataFrame({"id": [4, 5], "x": [4.0, 5.0], "y": [None, None]})
    spec = parse_task(_task(tmp_path, train, test))
    assert spec.target == "y"


def test_float_zero_one_target_is_binary(tmp_path):
    # метки 0.0/1.0 (целые, записанные как float) — классификация, а не регрессия
    n = 40
    train = pd.DataFrame({"id": range(n), "x": range(n), "churn": [float(i % 2) for i in range(n)]})
    test = pd.DataFrame({"id": range(n, n + 5), "x": range(5)})
    spec = parse_task(_task(tmp_path, train, test, text="Предскажите, уйдёт ли клиент."))
    assert spec.task_type == TaskType.BINARY


def test_float_continuous_target_stays_regression(tmp_path):
    train = pd.DataFrame({"id": range(30), "x": range(30), "y": [i * 0.37 for i in range(30)]})
    test = pd.DataFrame({"id": range(30, 33), "x": range(3)})
    spec = parse_task(_task(tmp_path, train, test, text="Предскажите y."))
    assert spec.task_type == TaskType.REGRESSION
