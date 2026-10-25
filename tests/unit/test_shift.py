"""Сдвиг train → test (ml/shift.py): AUC «train против test» в profile.warnings и agent.log, решения не меняет (#294)."""

import re

import numpy as np
import pandas as pd
import pytest

import ml.shift as shift
from interfaces.schemas import DataProfile
from ml.shift import SHIFT_PREFIX, adversarial_shift, log_shift, shift_features
from ml.validation import choose_validation
from tests.unit.test_validation import _spec


def _data(shift_by=0.0, n=800, seed=0, with_id=True):
    """Признак x в test сдвинут на shift_by; rec — номер записи (в test продолжается), const — константа."""
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({"x": rng.normal(0, 1, n), "z": rng.normal(0, 1, n), "const": 1})
    test = pd.DataFrame({"x": rng.normal(shift_by, 1, n // 2), "z": rng.normal(0, 1, n // 2), "const": 1})
    if with_id:
        train["rec"], test["rec"] = np.arange(n), np.arange(n, n + n // 2)
    train["y"] = rng.normal(size=n)
    return train, test


def _profile(columns, warnings=None) -> DataProfile:
    return DataProfile(
        n_train=0, n_test=0, numeric_features=list(columns), categorical_features=[], warnings=warnings or []
    )


def test_shift_found_and_top_feature_named():
    train, test = _data(shift_by=2.0)
    auc, top = adversarial_shift(_profile(["x", "z", "rec", "const"]), train, test)
    assert auc > 0.85 and top[0] == "x"


def test_no_shift_gives_low_auc_and_ids_are_not_shift():
    train, test = _data()
    profile = _profile(["x", "z", "rec", "const"])
    assert "rec" not in shift_features(profile, train, test) and "const" in shift_features(profile, train, test)
    auc, _ = adversarial_shift(profile, train, test)
    assert auc < 0.6  # номер записи в test продолжается, но это идентификатор — AUC не 1.0


def test_float_unique_column_is_a_feature():
    """Дробные числа с разными значениями (площадь) — обычный признак, а не идентификатор."""
    train, test = _data()
    assert "x" in shift_features(_profile(["x"]), train, test)


def test_warning_format_is_machine_readable():
    text = shift.shift_warning(0.884, ["a", "b"])
    assert re.match(r"^adversarial_auc=([0-9.]+)", text).group(1) == "0.88" and "a, b" in text
    quiet = shift.shift_warning(0.52, ["feat_q"])
    assert "нет" in quiet and "feat_q" not in quiet
    assert "без колонки t" in shift.shift_warning(0.5, [], excluded="t")


def test_time_column_excluded_from_warning_auc(caplog):
    """По колонке времени test отделяется всегда — в warnings AUC без неё, полный — в лог."""
    train, test = _data()
    train["t"], test["t"] = np.repeat(np.arange(8), 100), np.repeat([8, 9], 200)
    profile = _profile(["x", "z", "t"])
    log_shift(profile, train, test, "time:t")
    (line,) = profile.warnings
    assert float(re.match(r"^adversarial_auc=([0-9.]+)", line).group(1)) < 0.6 and "без колонки t" in line
    full = adversarial_shift(profile, train, test)[0]
    assert full > 0.95  # с колонкой времени — почти 1.0


def test_log_shift_is_idempotent_and_can_be_switched_off(monkeypatch):
    train, test = _data(shift_by=2.0)
    profile = _profile(["x", "z"])
    log_shift(profile, train, test, "kfold")
    log_shift(profile, train, test, "kfold")
    assert len(profile.warnings) == 1 and profile.warnings[0].startswith(SHIFT_PREFIX)
    off = _profile(["x", "z"])
    monkeypatch.setenv(shift.SHIFT_ENV, "0")
    log_shift(off, train, test, "kfold")
    assert off.warnings == []


def test_shift_error_never_breaks_the_agent(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("нет памяти")

    monkeypatch.setattr(shift, "adversarial_shift", boom)
    profile = _profile(["x"])
    log_shift(profile, *_data(), "kfold")
    assert profile.warnings == []


def test_choose_validation_scheme_unchanged_by_shift(monkeypatch):
    """Ступень «сдвиг» только пишет: схема та же с замером и без него."""
    train, test = _data(shift_by=2.0)
    with_shift = choose_validation(_spec(), _profile(["x", "z"]), train, test)
    monkeypatch.setenv(shift.SHIFT_ENV, "0")
    assert choose_validation(_spec(), _profile(["x", "z"]), train, test) == with_shift


@pytest.mark.parametrize("rows", [5, 12])
def test_too_few_rows_give_no_result(rows):
    train, test = _data()
    assert adversarial_shift(_profile(["x"]), train.head(rows), test.head(rows)) is None
