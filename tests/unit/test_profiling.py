"""Матрица признаков: категории из test, которых не было в train."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from interfaces.schemas import DataProfile
from ml.profiling import make_matrix


def _profile() -> DataProfile:
    return DataProfile(n_train=6, n_test=3, numeric_features=["num"], categorical_features=["city"])


def test_unseen_test_category_keeps_shared_vocabulary():
    train = pd.DataFrame({"num": [1, 2, 3, 4, 5, 6], "city": ["A", "B", "A", "B", "A", "B"]})
    test = pd.DataFrame({"num": [1, 2, 3], "city": ["A", "C", "B"]})  # "C" есть только в test

    x_train, x_test = make_matrix(_profile(), train, test)

    # Новая категория не потерялась и не стала пропуском
    assert x_test["city"].isna().sum() == 0
    assert x_test["city"].tolist() == ["A", "C", "B"]
    # Словарь общий: одно и то же значение получает один и тот же код в train и test
    assert list(x_train["city"].cat.categories) == list(x_test["city"].cat.categories)
    assert "C" in x_train["city"].cat.categories
    a_code = x_train["city"].cat.codes[x_train["city"] == "A"].iloc[0]
    assert x_test["city"].cat.codes.iloc[0] == a_code
    # Числа — float32 (экономия памяти)
    assert x_train["num"].dtype == np.float32 and x_test["num"].dtype == np.float32


def test_model_predicts_on_unseen_test_category():
    rng = np.random.default_rng(42)
    train = pd.DataFrame({"num": rng.normal(size=200), "city": rng.choice(["A", "B"], size=200)})
    y = train["num"] + (train["city"] == "A") * 3
    test = pd.DataFrame({"num": [0.0, 0.0, 0.0], "city": ["A", "B", "C"]})

    x_train, x_test = make_matrix(_profile(), train, test)
    model = lgb.LGBMRegressor(n_estimators=20, min_child_samples=5, verbose=-1).fit(x_train, y)
    pred = model.predict(x_test)

    assert pred.shape == (3,)
    assert np.isfinite(pred).all()


def test_special_characters_in_column_names_are_replaced():
    cols = ["income[usd]", "age<30", 'name"q', "income usd", "Площадь, м2"]
    train = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in cols})
    profile = DataProfile(n_train=3, n_test=3, numeric_features=cols, categorical_features=[])

    x_train, x_test = make_matrix(profile, train, train)

    assert list(x_train.columns) == ["income_usd", "age_30", "name_q", "income_usd_1", "Площадь_м2"]
    assert list(x_test.columns) == list(x_train.columns)


def test_load_data_keeps_string_ids_and_drops_missing_target(tmp_path):
    from interfaces.schemas import MetricSpec, TaskSpec, TaskType
    from ml.profiling import load_data

    (tmp_path / "train.csv").write_text("id,x,y\n001,1,5\n002,2,\n003,3,7\n")
    (tmp_path / "test.csv").write_text("id,x\n010,4\n")
    spec = TaskSpec(
        task_md=tmp_path / "task.md",
        task_dir=tmp_path,
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        submission_path=tmp_path / "submission.csv",
        task_type=TaskType.REGRESSION,
        target="y",
        id_column="id",
        metric=MetricSpec(name="rmse", greater_is_better=False),
        submission_columns=["id", "y"],
    )

    train, test = load_data(spec)

    assert train["id"].tolist() == ["001", "003"]  # строка с пустым target выкинута
    assert test["id"].tolist() == ["010"]  # ведущий ноль на месте


def _skewed_spec(metric: str):
    from pathlib import Path

    from interfaces.schemas import MetricSpec, TaskSpec, TaskType

    root = Path("/nonexistent")
    return TaskSpec(
        task_md=root / "task.md",
        task_dir=root,
        train_path=root / "train.csv",
        test_path=root / "test.csv",
        submission_path=root / "submission.csv",
        task_type=TaskType.REGRESSION,
        target="y",
        id_column="id",
        metric=MetricSpec(name=metric, greater_is_better=False),
        submission_columns=["id", "y"],
    )


def test_skew_warning_depends_on_metric():
    """Совет «log1p» — только для метрик в логарифмах; на RMSE — предупреждение, а не рекомендация (LLM его читает)."""
    from ml.profiling import profile_data

    rng = np.random.default_rng(0)
    train = pd.DataFrame({"id": range(500), "x": rng.normal(size=500), "y": rng.lognormal(sigma=1.5, size=500)})
    test = train.drop(columns="y")

    rmse_warn = " ".join(profile_data(_skewed_spec("rmse"), train, test).warnings)
    rmsle_warn = " ".join(profile_data(_skewed_spec("rmsle"), train, test).warnings)

    assert "skew=" in rmse_warn and "исходной шкале" in rmse_warn and "стоит попробовать" not in rmse_warn
    assert "совпадает" in rmsle_warn
