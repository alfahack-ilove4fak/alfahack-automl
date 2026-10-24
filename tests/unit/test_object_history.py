"""История объекта (feature_ops: object_history в ml/features.py): поиск ключа по данным и отсутствие утечки."""

import numpy as np
import pandas as pd
import pytest

from ml.features import ObjectHistory, apply_feature_ops, object_history_candidates, parse_feature_ops


def _panel(n_objects: int = 60, n_periods: int = 5, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Объекты × периоды: площадь и тип постоянны внутри объекта, погода меняется каждую строку."""
    rng = np.random.default_rng(seed)
    obj = np.repeat(np.arange(n_objects), n_periods)
    area = rng.uniform(100, 10_000, n_objects).round(1)
    kind = rng.choice(["office", "shop", "school"], n_objects)
    x = pd.DataFrame(
        {
            "Year_Factor": np.tile(np.arange(1, n_periods + 1), n_objects).astype(np.float32),
            "floor_area": area[obj].astype(np.float32),
            "facility_type": pd.Categorical(kind[obj]),
            "temperature": rng.normal(size=len(obj)).astype(np.float32),
        }
    )
    level = rng.normal(0, 10, n_objects)
    y = level[obj] + rng.normal(size=len(obj))
    return x, y


def test_candidates_find_object_key_and_time_from_data():
    x, _ = _panel()
    best = object_history_candidates(x)[0]
    assert best["time"] == "Year_Factor"
    assert "floor_area" in best["key"]
    assert "temperature" not in best["key"]  # меняется каждую строку — не повторяется из периода в период


def test_history_uses_only_past_periods_of_same_object():
    x, y = _panel()
    out = ObjectHistory().fit(x, y, classification=False).transform(x)
    first_obj = np.arange(5)  # строки объекта 0: периоды 1..5
    assert np.isnan(out["hist__mean__mean"].iloc[0])  # первый период — истории нет
    assert out["hist__count"].iloc[0] == 0
    for i in first_obj[1:]:
        assert out["hist__mean__mean"].iloc[i] == pytest.approx(y[:i].mean(), rel=1e-5)
        assert out["hist__mean__last"].iloc[i] == pytest.approx(y[i - 1], rel=1e-5)
        assert out["hist__count"].iloc[i] == i
    assert (out["hist__since_last"].dropna() == 1).all()


def test_history_does_not_see_current_row_or_future():
    """Меняем target текущего и будущих периодов — признаки строки не меняются."""
    x, y = _panel()
    base = ObjectHistory().fit(x, y, classification=False).transform(x)
    period = x["Year_Factor"].to_numpy()
    y_changed = np.where(period >= 3, y + 1000, y)
    changed = ObjectHistory().fit(x, y_changed, classification=False).transform(x)
    rows = period <= 3
    pd.testing.assert_frame_equal(base[rows], changed[rows])


def test_new_rows_get_history_from_fit_part_only():
    """Test — следующий период: история из train того же объекта, сам test в признаки не попадает."""
    x, y = _panel(n_periods=6)
    tr = x["Year_Factor"] <= 5
    x_tr, x_te = x[tr], x[~tr]
    _, te_out = apply_feature_ops(["object_history"], x_tr, y[tr.to_numpy()], False, x_te)
    obj0 = x_tr["floor_area"] == x_te["floor_area"].iloc[0]
    assert te_out["hist__mean__mean"].iloc[0] == pytest.approx(y[tr.to_numpy()][obj0.to_numpy()].mean(), rel=1e-5)
    assert te_out["hist__count"].iloc[0] == 5


def test_no_key_adds_nothing():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})  # нет ни времени, ни объектов
    out = ObjectHistory().fit(x, rng.normal(size=100), classification=False).transform(x)
    assert out.shape == (100, 0)
    assert object_history_candidates(x) == []


def test_explicit_key_and_time_and_classification():
    x, y = _panel()
    labels = (y > 0).astype(int)
    op = ObjectHistory(key=["floor_area"], time="Year_Factor").fit(x, labels, classification=True)
    out = op.transform(x)
    assert op.key == ["floor_area"]
    assert {"hist__p__mean", "hist__p__last", "hist__count", "hist__since_last"} <= set(out.columns)


def test_bad_params_fail_before_training():
    with pytest.raises(ValueError):
        parse_feature_ops([{"op": "object_history", "key": []}])
    with pytest.raises(TypeError):
        parse_feature_ops([{"op": "object_history", "window": 3}])
    x, y = _panel()
    with pytest.raises(ValueError):
        ObjectHistory(key=["missing"], time="Year_Factor").fit(x, y, classification=False)
