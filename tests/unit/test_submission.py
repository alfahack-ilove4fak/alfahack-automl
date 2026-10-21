"""validate_submission: ответ «колонка на класс» (вероятности по классам, #110)."""

import pandas as pd
import pytest

from evaluation.submission import per_class_columns, validate_submission
from interfaces.schemas import MetricSpec, TaskSpec, TaskType


def _spec(columns, target="label", task_type=TaskType.MULTICLASS, metric="logloss"):
    return TaskSpec(
        task_md="t/task.md",
        task_dir="t",
        train_path="t/train.csv",
        test_path="t/test.csv",
        submission_path="t/submission.csv",
        task_type=task_type,
        target=target,
        id_column="id",
        metric=MetricSpec(name=metric, greater_is_better=False),
        submission_columns=columns,
        class_labels=["a", "b", "c"],
    )


TEST = pd.DataFrame({"id": [1, 2, 3], "x": [0.1, 0.2, 0.3]})
PER_CLASS = _spec(["id", "a", "b", "c"])


def _check(tmp_path, rows, spec=PER_CLASS):
    path = tmp_path / "submission.csv"
    pd.DataFrame(rows, columns=spec.submission_columns).to_csv(path, index=False)
    return validate_submission(path, spec, TEST)


def test_per_class_columns_detection():
    assert per_class_columns(PER_CLASS) == ["a", "b", "c"]
    assert per_class_columns(_spec(["id", "label"])) == []
    # Колонок больше двух, но target среди них — это не «колонка на класс»
    assert per_class_columns(_spec(["id", "extra", "label"])) == []


def test_valid_probabilities(tmp_path):
    report = _check(tmp_path, [[1, 0.2, 0.3, 0.5], [2, 1.0, 0.0, 0.0], [3, 1 / 3, 1 / 3, 1 / 3]])
    assert report.ok, report.problems


@pytest.mark.parametrize(
    "rows, problem",
    [
        ([[1, 0.2, 0.3, 0.6], [2, 1, 0, 0], [3, 0.3, 0.3, 0.4]], "Сумма вероятностей"),
        ([[1, 1.2, -0.2, 0.0], [2, 1, 0, 0], [3, 0.3, 0.3, 0.4]], "вне отрезка"),
        ([[1, None, 0.5, 0.5], [2, 1, 0, 0], [3, 0.3, 0.3, 0.4]], "пропуски"),
        ([[1, "a", 0.5, 0.5], [2, 1, 0, 0], [3, 0.3, 0.3, 0.4]], "нечисловые"),
    ],
)
def test_bad_probabilities_are_caught(tmp_path, rows, problem):
    report = _check(tmp_path, rows)
    assert not report.ok and any(problem in p for p in report.problems), report.problems


def test_one_column_answer_still_checked_as_before(tmp_path):
    spec = _spec(["id", "label"], metric="f1_macro")
    report = _check(tmp_path, [[1, "a"], [2, "zzz"], [3, "c"]], spec)
    assert any("Метки вне списка классов" in p for p in report.problems)
