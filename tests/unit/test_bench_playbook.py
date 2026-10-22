"""Стенд playbook: честная отложенная часть — по времени (период или дата строкой) или по группам дубликатов."""

import numpy as np
import pandas as pd

from interfaces.schemas import DataProfile, Hypothesis, HypothesisKind, MetricSpec, TaskSpec, TaskType
from research.bench_playbook import honest_split, summarize, time_column, with_validation


def _spec(task_type=TaskType.REGRESSION):
    return TaskSpec(
        task_md="t/task.md",
        task_dir="t",
        train_path="t/train.csv",
        test_path="t/test.csv",
        submission_path="t/submission.csv",
        task_type=task_type,
        target="y",
        id_column="id",
        metric=MetricSpec(name="rmse", greater_is_better=False),
        submission_columns=["id", "y"],
    )


def _profile(numeric, categorical=()):
    return DataProfile(n_train=0, n_test=0, numeric_features=list(numeric), categorical_features=list(categorical))


def test_integer_period_after_train():
    train = pd.DataFrame({"year": [1, 1, 2, 2, 3, 3], "x": range(6), "y": range(6)})
    test = pd.DataFrame({"year": [4, 4], "x": [0, 1]})
    assert time_column(train, test, ["year", "x"]) == ("period", "year")
    tr, va, how = honest_split(_spec(), _profile(["year", "x"]), train, test)
    assert set(train.year.iloc[va]) == {3} and set(train.year.iloc[tr]) == {1, 2}
    assert how.startswith("время: year")


def test_date_strings_after_train():
    dates = pd.date_range("2024-01-01", periods=100, freq="D").strftime("%Y-%m-%d")
    train = pd.DataFrame({"date": dates, "store": ["a", "b"] * 50, "y": range(100)})
    test = pd.DataFrame({"date": ["2024-06-01", "2024-06-02"], "store": ["a", "b"]})
    assert time_column(train, test, ["date", "store"]) == ("date", "date")
    tr, va, _ = honest_split(_spec(), _profile([], ["date", "store"]), train, test)
    # в честной части — последние ~20% по дате, и все они позже train-части
    assert 15 <= len(va) <= 25
    assert pd.to_datetime(train.date.iloc[va]).min() > pd.to_datetime(train.date.iloc[tr]).max()


def test_dates_overlapping_test_are_not_time():
    dates = pd.date_range("2024-01-01", periods=50, freq="D").strftime("%Y-%m-%d")
    train = pd.DataFrame({"date": dates, "y": range(50)})
    test = pd.DataFrame({"date": dates[:5]})
    assert time_column(train, test, ["date"]) is None


def test_text_is_not_a_date():
    train = pd.DataFrame({"city": ["moscow", "kazan"] * 10, "y": range(20)})
    test = pd.DataFrame({"city": ["moscow"]})
    assert time_column(train, test, ["city"]) is None


def test_duplicates_stay_on_one_side():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 30, 300)
    train = pd.DataFrame({"x": x, "y": x % 2})
    test = pd.DataFrame({"x": [0, 1]})
    tr, va, how = honest_split(_spec(TaskType.BINARY), _profile(["x"]), train, test)
    assert how == "группы дубликатов"
    assert not set(train.x.iloc[tr]) & set(train.x.iloc[va])


def test_with_validation_like_agent():
    hyp = Hypothesis(id="h", kind=HypothesisKind.MODEL, title="h", params={"model": "xgboost"})
    assert with_validation(hyp, "time:year").params == {"model": "xgboost", "validation": "time:year"}
    assert "validation" not in with_validation(hyp, "kfold").params  # обычная CV — как без схемы
    own = hyp.model_copy(update={"params": {"validation": "group:x"}})
    assert with_validation(own, "time:year").params["validation"] == "group:x"
    assert hyp.params == {"model": "xgboost"}  # исходные params не тронуты


def test_summarize_counts_honest_wins_and_agent_decisions():
    rows = pd.DataFrame(
        [
            {"task": "a", "id": "baseline", "honest": 10.0, "seconds": 1},
            {"task": "a", "id": "h", "honest": 9.0, "honest_gain": 1.0, "accepted": True, "seconds": 2},
            {"task": "b", "id": "baseline", "honest": 2.0, "seconds": 1},
            {"task": "b", "id": "h", "honest": 2.5, "honest_gain": -0.5, "accepted": True, "seconds": 4},
            {"task": "b", "id": "baseline", "honest": 2.0, "seconds": 1},  # второй CSV: тот же бейзлайн
        ]
    )
    row = summarize(rows).loc["h"]
    assert (row.tasks, row.wins, row.accepted, row.accepted_and_win) == (2, 1, 2, 1)
    assert row.win_share == 0.5
    assert np.isclose(row.median_gain_pct, (10 - 25) / 2)


def test_summarize_orders_by_win_share_then_median():
    rows = pd.DataFrame(
        [{"task": t, "id": "baseline", "honest": 1.0, "seconds": 1} for t in "abc"]
        + [
            {"task": t, "id": "wide", "honest_gain": g, "seconds": 1}
            for t, g in zip("abc", [0.01, 0.01, -0.5], strict=True)
        ]
        + [{"task": "a", "id": "narrow", "honest_gain": 0.2, "seconds": 1}]  # 1/1 задач — выше, чем 2/3
    )
    assert list(summarize(rows).index) == ["narrow", "wide"]
