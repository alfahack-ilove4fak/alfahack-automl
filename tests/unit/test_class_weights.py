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
