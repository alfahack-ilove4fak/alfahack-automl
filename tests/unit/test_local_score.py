"""Локальный балл: формула организаторов и пересчёт submission.csv во вход метрики."""

import numpy as np
import pandas as pd
import pytest

from evaluation.local_score import organizer_score, score_submission
from interfaces.schemas import MetricSpec, TaskSpec, TaskType


def _spec(task_type, metric, greater, labels=None):
    return TaskSpec(
        task_md="t/task.md",
        task_dir="t",
        train_path="t/train.csv",
        test_path="t/test.csv",
        submission_path="t/submission.csv",
        task_type=task_type,
        target="y",
        id_column="id",
        metric=MetricSpec(name=metric, greater_is_better=greater),
        submission_columns=["id", "y"],
        class_labels=labels,
    )


@pytest.mark.parametrize(
    "metric, value, baseline, expected",
    [
        ("rmse", 50, 100, 50.0),  # 100 · (1 − 50/100)
        ("rmse", 120, 100, 0.0),  # хуже бейзлайна — ноль, а не минус
        ("f1_macro", 0.6, 0.2, 50.0),  # 100 · (0.6 − 0.2) / (1 − 0.2)
        ("f1_macro", 0.1, 0.2, 0.0),
        ("roc_auc", 0.75, 0.5, 50.0),
        ("accuracy", 1.0, 1.0, 0.0),  # бейзлайн уже идеален — расти некуда
        ("rmse", 1.0, 0.0, 0.0),
        ("rmse", float("nan"), 1.0, 0.0),
    ],
)
def test_organizer_score(metric, value, baseline, expected):
    assert organizer_score(metric, value, baseline) == pytest.approx(expected)


def test_regression_perfect_answer_scores_100():
    spec = _spec(TaskType.REGRESSION, "rmse", False)
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = score_submission(spec, y, y, y_train=pd.Series([1.0, 5.0]))
    assert result["value"] == 0 and result["baseline"] > 0 and result["score"] == 100


def test_labels_are_encoded_like_ml():
    spec = _spec(TaskType.MULTICLASS, "f1_macro", True, labels=["a", "b", "c"])
    answers = pd.Series(["a", "b", "c", "c"])
    result = score_submission(spec, answers, answers.copy(), y_train=pd.Series(["c", "c", "a"]))
    assert result["value"] == 1.0 and result["score"] == 100


def test_binary_probability_and_trivial_baseline():
    spec = _spec(TaskType.BINARY, "roc_auc", True, labels=["no", "yes"])
    answers = pd.Series(["no", "yes", "no", "yes"])
    result = score_submission(spec, answers, pd.Series([0.1, 0.9, 0.2, 0.7]), y_train=answers)
    assert result["value"] == 1.0
    assert result["baseline"] == 0.5  # доля класса одинакова для всех строк — AUC 0.5


def test_trivial_answer_scores_zero():
    spec = _spec(TaskType.MULTICLASS, "accuracy", True, labels=["a", "b"])
    answers = pd.Series(["a", "a", "b"])
    result = score_submission(spec, answers, pd.Series(["a", "a", "a"]), y_train=pd.Series(["a", "a", "b"]))
    assert result["score"] == 0


def test_unknown_label_in_submission_is_an_error():
    spec = _spec(TaskType.MULTICLASS, "accuracy", True, labels=["a", "b"])
    with pytest.raises(ValueError, match="вне списка классов"):
        score_submission(spec, pd.Series(["a"]), pd.Series(["zzz"]), y_train=pd.Series(["a"]))


def test_multiclass_proba_metric_accepts_labels():
    """Для многих классов с logloss агент пишет метку — считаем её «уверенной» вероятностью."""
    spec = _spec(TaskType.MULTICLASS, "logloss", False, labels=["a", "b", "c"])
    answers = pd.Series(["a", "b", "c"])
    result = score_submission(spec, answers, answers.copy(), y_train=answers)
    assert np.isfinite(result["value"]) and result["value"] < result["baseline"]
