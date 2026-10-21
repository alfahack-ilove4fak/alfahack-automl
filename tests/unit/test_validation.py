"""Схемы кросс-валидации (ml/validation.py) и выбор схемы по умолчанию."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interfaces.schemas import DataProfile, MetricSpec, TaskSpec, TaskType
from ml.experiment import _split
from ml.validation import choose_validation, make_folds, parse_validation, time_values


def _spec(task_type=TaskType.REGRESSION, labels=None) -> TaskSpec:
    root = Path("/nonexistent")
    metric = MetricSpec(name="rmse", greater_is_better=False)
    if task_type != TaskType.REGRESSION:
        metric = MetricSpec(name="f1_macro", greater_is_better=True)
    return TaskSpec(
        task_md=root / "task.md",
        task_dir=root,
        train_path=root / "train.csv",
        test_path=root / "test.csv",
        submission_path=root / "submission.csv",
        task_type=task_type,
        target="y",
        id_column="id",
        metric=metric,
        submission_columns=["id", "y"],
        class_labels=labels,
    )


def _profile(numeric, categorical=()) -> DataProfile:
    return DataProfile(n_train=0, n_test=0, numeric_features=list(numeric), categorical_features=list(categorical))


@pytest.mark.parametrize(
    "value, task_type, expected",
    [
        (None, TaskType.REGRESSION, ("kfold", None)),
        (None, TaskType.BINARY, ("stratified", None)),
        ("stratified", TaskType.REGRESSION, ("kfold", None)),
        ("time:year", TaskType.REGRESSION, ("time", "year")),
        ("Group : duplicates", TaskType.BINARY, ("group", "duplicates")),
    ],
)
def test_parse_validation(value, task_type, expected):
    assert parse_validation(value, _spec(task_type, ["a", "b"])) == expected


@pytest.mark.parametrize("value", ["random", "time:", 5])
def test_parse_validation_rejects_garbage(value):
    with pytest.raises(ValueError):
        parse_validation(value, _spec())


def test_time_values_numbers_dates_and_not_time():
    assert time_values(pd.Series([2001, 2002, 2003])).tolist() == [2001, 2002, 2003]
    dates = time_values(pd.Series(["2020-01-05", "2020-02-01", "2019-12-31"]))
    assert dates is not None and dates[2] < dates[0] < dates[1]
    assert time_values(pd.Series(["Москва", "Казань", "Сочи"])) is None


def test_time_folds_validate_only_on_later_periods():
    n = 600
    train = pd.DataFrame({"year": np.repeat([1, 2, 3, 4, 5, 6], n // 6), "x": np.arange(n)})
    y = np.arange(n, dtype=float)
    folds = make_folds(_spec(), "time", "year", train, ["year", "x"], y, 5, _split)
    assert len(folds) == 5
    for tr, va in folds:
        assert train["year"].iloc[tr].max() < train["year"].iloc[va].min()


def test_duplicates_never_split_between_train_and_valid():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"a": rng.integers(0, 20, 300), "b": rng.integers(0, 3, 300)})
    y = rng.choice(2, size=300)
    spec = _spec(TaskType.BINARY, ["0", "1"])
    for tr, va in make_folds(spec, "group", "duplicates", base, ["a", "b"], y, 5, _split):
        keys_tr = set(map(tuple, base.iloc[tr].to_numpy()))
        keys_va = set(map(tuple, base.iloc[va].to_numpy()))
        assert not keys_tr & keys_va


def test_choose_validation_time_from_date_strings():
    days = pd.date_range("2020-01-01", periods=400, freq="D").strftime("%Y-%m-%d")
    train = pd.DataFrame({"sold": days[:300], "x": np.arange(300.0)})
    test = pd.DataFrame({"sold": days[300:], "x": np.arange(100.0)})
    assert choose_validation(_spec(), _profile(["x"], ["sold"]), train, test) == "time:sold"


def test_choose_validation_duplicates_then_default():
    rng = np.random.default_rng(1)
    dup = pd.DataFrame({"a": rng.integers(0, 5, 200), "b": rng.integers(0, 2, 200)})
    assert choose_validation(_spec(), _profile(["a", "b"]), dup, dup.head(10)) == "group:duplicates"
    uniq = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    test = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    assert choose_validation(_spec(TaskType.BINARY, ["0", "1"]), _profile(["a", "b"]), uniq, test) == "stratified"


def test_choose_validation_ignores_almost_constant_flag():
    """Флаг, почти всегда равный 0 и в train, и в test, — не время (на public/multiclass так выбирался is_ftp_login)."""
    rng = np.random.default_rng(2)
    flag_train = np.where(rng.random(500) < 0.02, rng.integers(1, 4, 500), 0)
    flag_test = np.zeros(200, dtype=int)
    train = pd.DataFrame({"flag": flag_train, "x": rng.normal(size=500)})
    test = pd.DataFrame({"flag": flag_test, "x": rng.normal(size=200)})
    assert not choose_validation(_spec(), _profile(["flag", "x"]), train, test).startswith("time")
