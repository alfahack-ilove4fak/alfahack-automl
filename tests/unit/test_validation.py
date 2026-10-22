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


def _rare_event(n_months=6, per_month=200, seed=3):
    """Бинарная задача с редким событием по месяцам; в месяцах 1 и 3 событий нет совсем (ревью #150)."""
    rng = np.random.default_rng(seed)
    month = np.repeat(np.arange(1, n_months + 1), per_month)
    x = rng.normal(size=len(month))
    y = np.where((x > 2.0) & ~np.isin(month, [1, 3]), "1", "0")
    y[np.flatnonzero(~np.isin(month, [1, 3]))[::40]] = "1"  # в остальных месяцах события точно есть
    return pd.DataFrame({"month": month, "x": x, "y": y})


def test_time_folds_drop_periods_with_one_class():
    train = _rare_event()
    spec = _spec(TaskType.BINARY, ["0", "1"])
    folds = make_folds(spec, "time", "month", train, ["month", "x"], train["y"].to_numpy(), 5, _split)
    assert folds
    for tr, va in folds:
        assert train["y"].iloc[tr].nunique() == 2 and train["y"].iloc[va].nunique() == 2


def test_rare_event_with_time_scheme_does_not_fail():
    """Классификация + time: период без положительных не роняет гипотезу (ROC-AUC на одном классе не считается)."""
    from interfaces.schemas import Hypothesis, HypothesisKind, MetricSpec, RunBudget
    from ml.experiment import run_experiment

    train = _rare_event()
    spec = _spec(TaskType.BINARY, ["0", "1"]).model_copy(
        update={"metric": MetricSpec(name="roc_auc", greater_is_better=True)}
    )
    hyp = Hypothesis(
        id="h",
        kind=HypothesisKind.MODEL,
        title="тест",
        params={"validation": "time:month", "model_params": {"n_estimators": 20}},
    )
    result = run_experiment(spec, _profile(["month", "x"]), hyp, train, RunBudget())
    assert result.status.value == "ok", result.error


def test_choose_validation_skips_time_without_two_working_folds():
    """Событие только в последнем месяце: у time нет ни одного фолда с двумя классами → другая схема."""
    rng = np.random.default_rng(4)
    month = np.repeat(np.arange(1, 7), 100)
    train = pd.DataFrame({"month": month, "x": rng.normal(size=600)})
    train["y"] = np.where((month == 6) & (train["x"] > 0), "1", "0")
    test = pd.DataFrame({"month": np.full(50, 7), "x": rng.normal(size=50)})
    chosen = choose_validation(_spec(TaskType.BINARY, ["0", "1"]), _profile(["month", "x"]), train, test)
    assert not chosen.startswith("time")


def test_choose_validation_row_number_is_not_time():
    """Номер строки на случайном разбиении «продолжается» в test, но это не время (ревью #150)."""
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"record_no": np.arange(400), "x": rng.normal(size=400)})
    test = pd.DataFrame({"record_no": np.arange(400, 500), "x": rng.normal(size=100)})
    assert choose_validation(_spec(), _profile(["record_no", "x"]), train, test) == "kfold"
    # те же номера, но имя «про время» — время
    named = train.rename(columns={"record_no": "day_index"})
    named_test = test.rename(columns={"record_no": "day_index"})
    assert choose_validation(_spec(), _profile(["day_index", "x"]), named, named_test) == "time:day_index"


def test_time_name_hint_needs_word_start():
    from ml.validation import TIME_NAME_HINT

    assert TIME_NAME_HINT.search("Year_Factor") and TIME_NAME_HINT.search("sale_date")
    assert not TIME_NAME_HINT.search("update_count")


def test_duplicate_groups_without_text_conversion_of_numbers():
    """Группы те же, что при astype(str): числа, NaN, категории и object со смешанными типами."""
    from ml.validation import duplicate_groups

    df = pd.DataFrame(
        {
            "a": [1.0, 1.0, np.nan, np.nan, 2.0],
            "c": pd.Categorical(["x", "x", "y", "y", "x"]),
            "o": ["s", "s", 3, 3, "s"],
        }
    )
    groups = duplicate_groups(df)
    assert groups[0] == groups[1] and groups[2] == groups[3]
    assert len(set(groups)) == 3
    reference = pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy()
    assert (pd.Series(groups).factorize()[0] == pd.Series(reference).factorize()[0]).all()
