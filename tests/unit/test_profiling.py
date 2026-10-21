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
