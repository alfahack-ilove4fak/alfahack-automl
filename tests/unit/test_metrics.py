"""Реестр метрик: каждая метрика находится по своим именам и считается на данных в формате ml/experiment.py.

Метки там закодированы числами 0..k-1, вероятности — матрица (n, k). Если метрика падает на таком входе,
гипотеза на скрытой задаче с этой метрикой получит FAILED.
"""

import numpy as np
import pytest

from evaluation.metrics import METRICS, SEARCH_ORDER, compute_metric, find_metric_in_text

RNG = np.random.default_rng(0)


def test_search_order_covers_registry():
    assert sorted(SEARCH_ORDER) == sorted(METRICS)
    assert len(SEARCH_ORDER) == len(set(SEARCH_ORDER))


@pytest.mark.parametrize("name, alias", [(m.name, a) for m in METRICS.values() for a in m.aliases])
def test_every_alias_finds_its_metric(name, alias):
    # «*» — любое окончание, в том числе пустое
    assert find_metric_in_text(f"Метрика: {alias.replace('*', '')}") == name


def _inputs(prediction: str, n_classes: int):
    """Идеальный и зашумлённый ответ в формате, который ml/experiment.py передаёт в метрику."""
    n = 200
    if prediction == "value":
        y = RNG.uniform(1, 100, n)
        return y, y.copy(), y + RNG.normal(0, 20, n).clip(-0.9 * y)
    y = np.arange(n) % n_classes
    if prediction == "label":
        noisy = y.copy()
        noisy[: n // 2] = RNG.integers(0, n_classes, n // 2)
        return y, y.copy(), noisy
    perfect = np.eye(n_classes)[y] * 0.98 + 0.02 / n_classes
    noisy = RNG.dirichlet(np.ones(n_classes), n)
    return y, perfect, noisy


@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("name", sorted(METRICS))
def test_metric_works_on_ml_inputs(name, n_classes):
    metric = METRICS[name]
    if metric.prediction == "value" and n_classes == 4:
        pytest.skip("регрессии классы не нужны")
    y, perfect, noisy = _inputs(metric.prediction, n_classes)
    good, bad = compute_metric(name, y, perfect), compute_metric(name, y, noisy)
    assert np.isfinite(good) and np.isfinite(bad)
    assert (good > bad) if metric.greater_is_better else (good < bad)


def test_known_values():
    assert compute_metric("smape", [100, 0], [50, 0]) == pytest.approx(100 * (2 * 50 / 150) / 2)
    assert compute_metric("wape", [10, 30], [12, 24]) == pytest.approx(8 / 40)
    assert compute_metric("msle", [np.e - 1], [0]) == pytest.approx(1.0)
    y, proba = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert compute_metric("gini", y, proba) == pytest.approx(2 * compute_metric("roc_auc", y, proba) - 1)
    assert compute_metric("pr_auc", [0, 1, 1], np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])) == 1.0


def test_binary_metrics_do_not_crash_on_multiclass():
    """«F1» в task.md при 5 классах: считаем macro, а не падаем с ошибкой sklearn."""
    y, p = [0, 1, 2, 3, 4], [0, 1, 2, 3, 3]
    assert compute_metric("f1", y, p) == pytest.approx(compute_metric("f1_macro", y, p))
    assert 0 < compute_metric("precision", y, p) < 1
    assert 0 < compute_metric("recall", y, p) < 1
