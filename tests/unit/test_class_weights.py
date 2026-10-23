"""Множители вероятностей классов (ml/class_weights.py) под метрику по меткам."""

import numpy as np

from evaluation.metrics import compute_metric
from ml.class_weights import cross_fit_scores, fit_weights


def _underconfident_rare_class(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Класс 2 редкий (5%), и модель занижает его вероятность — argmax его почти не выбирает."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.6, 0.35, 0.05])
    logits = rng.normal(size=(n, 3))
    logits[np.arange(n), y] += 2.0
    logits[:, 2] -= 2.5  # недооценка редкого класса
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y, proba


def test_fit_weights_lifts_rare_class():
    y, proba = _underconfident_rare_class()
    before = compute_metric("f1_macro", y, proba.argmax(axis=1))
    w = fit_weights("f1_macro", True, y, proba)
    after = compute_metric("f1_macro", y, (proba * w).argmax(axis=1))
    assert after > before + 0.05
    assert w[2] > w[0]  # редкому классу множитель больше


def test_cross_fit_scores_improve_on_held_out_folds():
    y, proba = _underconfident_rare_class()
    folds = [(y[i::5], proba[i::5]) for i in range(5)]
    plain = [compute_metric("f1_macro", yk, pk.argmax(axis=1)) for yk, pk in folds]
    adjusted = cross_fit_scores("f1_macro", True, folds)
    assert all(a > p for a, p in zip(adjusted, plain, strict=True))


def test_cross_fit_does_not_invent_gain_on_noise():
    """Вероятности — шум: подгонка на тех же данных «находит» выигрыш, кросс-подгонка — нет."""
    rng = np.random.default_rng(1)
    y = rng.choice(3, size=1500)
    proba = rng.dirichlet(np.ones(3), size=1500)
    folds = [(y[i::5], proba[i::5]) for i in range(5)]
    plain = np.mean([compute_metric("f1_macro", yk, pk.argmax(axis=1)) for yk, pk in folds])
    in_sample = np.mean(
        [
            compute_metric("f1_macro", yk, (pk * fit_weights("f1_macro", True, yk, pk)).argmax(axis=1))
            for yk, pk in folds
        ]
    )
    honest = np.mean(cross_fit_scores("f1_macro", True, folds))
    assert in_sample > plain  # переобучение под свои же данные
    assert honest < in_sample
    assert abs(honest - plain) < 0.03


def test_grid_reaches_low_threshold_for_rare_class():
    """#191: у редкого класса лучший порог бывает < 0.06 — сетка должна давать отношение множителей > 16."""
    rng = np.random.default_rng(3)
    n = 20000
    y = (rng.random(n) < 0.01).astype(int)
    p1 = np.clip(np.where(y == 1, rng.normal(0.035, 0.01, n), rng.normal(0.01, 0.008, n)), 1e-4, 0.2)
    proba = np.column_stack([1 - p1, p1])
    w = fit_weights("f1", True, y, proba)
    assert w[1] / w[0] > 16
    assert _score_f1(y, proba, w) > _score_f1(y, proba, np.ones(2))


def _score_f1(y, proba, w):
    return compute_metric("f1", y, (proba * w).argmax(axis=1))


def test_fit_weights_subsamples_large_oof(monkeypatch):
    """Больше MAX_FIT_ROWS строк — подбор на детерминированной подвыборке: те же множители при повторе."""
    import ml.class_weights as cw

    monkeypatch.setattr(cw, "MAX_FIT_ROWS", 500)
    seen = []
    original = cw._score
    monkeypatch.setattr(cw, "_score", lambda metric, y, proba, w: seen.append(len(y)) or original(metric, y, proba, w))
    y, proba = _underconfident_rare_class(n=3000)
    first = cw.fit_weights("f1_macro", True, y, proba)
    second = cw.fit_weights("f1_macro", True, y, proba)
    assert set(seen) == {500}
    assert np.array_equal(first, second)
