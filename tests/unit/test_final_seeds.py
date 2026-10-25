"""Сиды финала (#261, E3): среднее нескольких копий модели с разными seed; AGENT_FINAL_SEEDS=1 — как раньше."""

import numpy as np
import pandas as pd
import pytest

from interfaces.schemas import DataProfile, Hypothesis, HypothesisKind, TaskType
from ml import experiment
from ml.experiment import SEED, final_seeds, fit_predict_final
from tests.unit.test_objective_metric import _skewed, _spec


def _hyp(**params) -> Hypothesis:
    params = {"model": "lightgbm", "_best_iteration": 40, **params}
    return Hypothesis(id="h", kind=HypothesisKind.HYPERPARAMS, title="сиды", params=params)


def _answer(monkeypatch, seeds, hyp, spec=None, data=None):
    monkeypatch.setenv(experiment.FINAL_SEEDS_ENV, str(seeds))
    train, test, profile = data or _skewed()
    return fit_predict_final(spec or _spec("rmse"), profile, hyp, train, test)


def test_final_seeds_env(monkeypatch):
    monkeypatch.delenv(experiment.FINAL_SEEDS_ENV, raising=False)
    # по умолчанию — по размеру таблицы: до 10 млн ячеек 5, до 25 млн 3, больше один
    assert [final_seeds(c) for c in (0, 10_000_000, 10_000_001, 25_000_000, 25_000_001)] == [5, 5, 3, 3, 1]
    for value, expected in [("1", 1), ("0", 1), ("-4", 1), ("4", 4), ("99", experiment.MAX_FINAL_SEEDS)]:
        monkeypatch.setenv(experiment.FINAL_SEEDS_ENV, value)
        assert final_seeds(10**9) == expected  # явное число — независимо от размера
    monkeypatch.setenv(experiment.FINAL_SEEDS_ENV, "мусор")
    assert final_seeds(100) == 5  # мусор — как не задано


def test_regression_mean_of_separate_seeds(monkeypatch):
    """Прогноз 3 сидов = среднее трёх отдельных финалов с seed SEED, SEED+1, SEED+2."""
    data = _skewed()
    parts = [
        _answer(monkeypatch, 1, _hyp(model_params={"random_state": SEED + i}), data=data)["y"].to_numpy()
        for i in range(3)
    ]
    averaged = _answer(monkeypatch, 3, _hyp(), data=data)["y"].to_numpy()
    np.testing.assert_allclose(averaged, np.mean(parts, axis=0), rtol=1e-9)
    assert not np.allclose(parts[0], parts[1])  # seed действительно меняет модель (subsample/colsample)


def test_one_seed_is_the_old_answer_byte_for_byte(monkeypatch):
    data = _skewed()
    one = _answer(monkeypatch, 1, _hyp(), data=data)
    explicit = _answer(monkeypatch, 1, _hyp(model_params={"random_state": SEED}), data=data)
    pd.testing.assert_frame_equal(one, explicit)


def test_explicit_seed_in_model_params_is_the_first(monkeypatch):
    data = _skewed()
    first = _answer(monkeypatch, 1, _hyp(model_params={"random_state": 7}), data=data)["y"].to_numpy()
    second = _answer(monkeypatch, 1, _hyp(model_params={"random_state": 8}), data=data)["y"].to_numpy()
    both = _answer(monkeypatch, 2, _hyp(model_params={"random_state": 7}), data=data)["y"].to_numpy()
    np.testing.assert_allclose(both, (first + second) / 2, rtol=1e-9)


def test_classification_mean_of_probabilities(monkeypatch):
    rng = np.random.default_rng(1)
    n = 600
    train = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    train["y"] = np.where(train.a + 0.7 * rng.normal(size=n) > 0, "yes", "no")
    test = train.drop(columns="y").head(150)
    profile = DataProfile(n_train=n, n_test=150, numeric_features=["a", "b"], categorical_features=[])
    spec = _spec("roc_auc", TaskType.BINARY).model_copy(update={"class_labels": ["no", "yes"]})
    data = (train, test, profile)
    hyp = lambda **mp: _hyp(model_params={"subsample": 0.6, **mp})  # noqa: E731 — подвыборка строк: seed значим
    parts = [_answer(monkeypatch, 1, hyp(random_state=SEED + i), spec, data)["y"].to_numpy() for i in range(3)]
    averaged = _answer(monkeypatch, 3, hyp(), spec, data)["y"].to_numpy()
    np.testing.assert_allclose(averaged, np.mean(parts, axis=0), rtol=1e-9)


def test_ensembles_and_deterministic_models_are_not_multiplied(monkeypatch):
    """Ансамбль уже усредняет, у hist_gb/linear в финале seed ничего не меняет — по одному обучению."""
    calls = []
    original = experiment._fit_model
    monkeypatch.setattr(experiment, "_fit_model", lambda *a, **kw: (calls.append(1), original(*a, **kw))[1])
    data = _skewed()
    for hyp in (_hyp(ensemble=["lightgbm", "hist_gb"], _best_iteration=[30, 30]), _hyp(model="hist_gb")):
        calls.clear()
        _answer(monkeypatch, 3, hyp, data=data)
        assert len(calls) == 1
    calls.clear()
    _answer(monkeypatch, 3, _hyp(), data=data)
    assert len(calls) == 3


def test_huge_tables_use_one_seed(monkeypatch):
    monkeypatch.delenv(experiment.FINAL_SEEDS_ENV, raising=False)
    monkeypatch.setattr(experiment, "FINAL_SEEDS_BY_CELLS", ((100, 5),))
    calls = []
    original = experiment._fit_model
    monkeypatch.setattr(experiment, "_fit_model", lambda *a, **kw: (calls.append(1), original(*a, **kw))[1])
    train, test, profile = _skewed()
    fit_predict_final(_spec("rmse"), profile, _hyp(), train, test)  # AGENT_FINAL_SEEDS не задан: таблица > 100 ячеек
    assert len(calls) == 1


@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_heavy_models_average_seeds(monkeypatch, model):
    pytest.importorskip(model)  # из группы full
    data = _skewed()
    params = {"model": model, "_best_iteration": 30}
    key = experiment.FINAL_SEED_KEYS[model]
    parts = [
        _answer(monkeypatch, 1, _hyp(**params, model_params={key: SEED + i}), data=data)["y"].to_numpy()
        for i in range(2)
    ]
    averaged = _answer(monkeypatch, 2, _hyp(**params), data=data)["y"].to_numpy()
    np.testing.assert_allclose(averaged, np.mean(parts, axis=0), rtol=1e-6)
