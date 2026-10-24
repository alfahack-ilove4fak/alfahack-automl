"""ml.profiling.leak_check (#240, чужая задача leak_column_auc): признаки, которых не будет на момент прогноза."""

from pathlib import Path

import numpy as np
import pandas as pd

from interfaces.schemas import MetricSpec, TaskSpec, TaskType
from ml.profiling import leak_check, profile_data


def _spec(task_type: TaskType = TaskType.BINARY) -> TaskSpec:
    root = Path("/nonexistent")
    return TaskSpec(
        task_md=root / "task.md",
        task_dir=root,
        train_path=root / "train.csv",
        test_path=root / "test.csv",
        submission_path=root / "submission.csv",
        task_type=task_type,
        target="y",
        id_column="id",
        metric=MetricSpec(name="roc_auc" if task_type == TaskType.BINARY else "rmse", greater_is_better=True),
        submission_columns=["id", "y"],
        class_labels=["0", "1"] if task_type == TaskType.BINARY else None,
    )


def _frames(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = (rng.random(n) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    df = pd.DataFrame({"id": range(n), "a": x[:, 0], "b": x[:, 1], "y": y})
    return df.iloc[: n // 2].reset_index(drop=True), df.iloc[n // 2 :].drop(columns="y").reset_index(drop=True)


def test_column_empty_in_test_is_dropped():
    """Как leak_column_auc: поле = target на 98% train, в test пусто."""
    train, test = _frames()
    rng = np.random.default_rng(1)
    train["status"] = np.where(rng.random(len(train)) < 0.98, train["y"], 1 - train["y"]).astype(float)
    test["status"] = np.nan
    drop, warnings = leak_check(_spec(), train, test, ["a", "b", "status"])
    assert drop == ["status"]
    assert "status" in warnings[0] and "пуст" in warnings[0]


def test_column_mostly_missing_in_both_is_kept():
    """Редко заполненный признак (пропуски и в train) — не утечка: в test он так же редок."""
    train, test = _frames()
    train["rare"] = np.where(np.arange(len(train)) % 10 == 0, 1.0, np.nan)
    test["rare"] = np.nan
    drop, _ = leak_check(_spec(), train, test, ["a", "rare"])
    assert drop == []


def test_strong_signal_with_shift_is_dropped():
    """Почти копия target в train, а в test — заглушка (нули вместо пропусков): та же утечка."""
    train, test = _frames()
    train["copy"] = train["y"] + np.random.default_rng(2).normal(scale=0.05, size=len(train))
    test["copy"] = 0.0
    drop, warnings = leak_check(_spec(), train, test, ["a", "copy"])
    assert drop == ["copy"]
    assert "KS" in warnings[0]


def test_strong_honest_feature_kept_with_one_warning():
    """Сильный признак с тем же распределением в test (как в cancer) — честный: оставляем, одно предупреждение."""
    train, test = _frames()
    for frame in (train, test):
        frame["strong"] = frame["a"] * 3
    train["y"] = (train["a"] > 0).astype(int)  # признак идеально разделяет классы в train
    drop, warnings = leak_check(_spec(), train, test, ["a", "b", "strong"])
    assert drop == []
    assert len(warnings) == 1 and "оставлены" in warnings[0] and "strong" in warnings[0]


def test_regression_uses_rank_correlation():
    train, test = _frames()
    train["y"] = train["a"] * 10 + 1
    train["leak"] = train["y"] * 2
    test["leak"] = -100.0
    spec = _spec(TaskType.REGRESSION)
    drop, _ = leak_check(spec, train, test, ["a", "b", "leak"])
    assert "leak" in drop


def test_empty_test_drops_nothing():
    train, test = _frames()
    drop, warnings = leak_check(_spec(), train, test.head(0), ["a", "b"])
    assert drop == [] and warnings == []


def test_profile_excludes_leak_and_records_warning():
    train, test = _frames()
    train["status"] = train["y"].astype(float)
    test["status"] = np.nan
    profile = profile_data(_spec(), train, test)
    assert "status" in profile.dropped_features
    assert "status" not in profile.numeric_features + profile.categorical_features
    assert any("status" in w for w in profile.warnings)
