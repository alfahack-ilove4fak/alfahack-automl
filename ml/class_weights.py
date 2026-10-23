"""Множители вероятностей классов под метрику по меткам (Macro F1, accuracy…) — params.proba_weights. Владелец — ml.

Ответ = argmax(вероятности × w): множитель > 1 делает класс «смелее», < 1 — «осторожнее». Для Macro F1 это поднимает
редкие классы, которые модель с argmax почти никогда не выбирает.

w подбирается только на OOF-вероятностях (предсказаниях фолдов на данных, которых модель не видела), не на test.
Честная оценка — кросс-подгонка: множители для фолда k подбираются на OOF остальных фолдов и применяются к фолду k,
поэтому скор по фолдам не видит собственной подгонки и сравним с другими гипотезами попарно.
"""

from __future__ import annotations

import numpy as np

from evaluation.metrics import compute_metric

# ×0.1 … ×10, в середине ×1. Для бинарной задачи отношение множителей — порог: ×0.25…×4 давал порог не ниже ≈ 0.06,
# а у редкого класса лучший порог бывает ниже (#191: XGBoost на payments — 0.07, на краю сетки)
GRID = np.exp(np.linspace(np.log(0.1), np.log(10.0), 25))
PASSES = 3
# Бинарная задача: множители (1, r) = порог 1 / (1 + r) по вероятности класса 1; перебираем r напрямую — покоординатный
# подъём застревает на плато (F1 = 0, пока порог выше всех вероятностей редкого класса) и не доходит до порога < 0.06
BINARY_RATIOS = np.exp(np.linspace(np.log(0.01), np.log(100.0), 81))  # порог от ≈ 0.01 до ≈ 0.99
MAX_FIT_ROWS = (
    50_000  # подбор на подвыборке OOF: с #191 он идёт в каждом эксперименте, на 500k строк × 10 классов — минуты
)
SEED = 42


def _score(metric: str, y: np.ndarray, proba: np.ndarray, w: np.ndarray) -> float:
    return compute_metric(metric, y, (proba * w).argmax(axis=1))


def fit_weights(metric: str, greater_is_better: bool, y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Покоординатный подъём по сетке GRID: по очереди для каждого класса — лучший множитель при остальных фиксированных.

    Строк больше MAX_FIT_ROWS — подбор на детерминированной подвыборке (множители — доли, им хватает 50 тыс. строк).
    """
    if len(y) > MAX_FIT_ROWS:
        rows = np.random.default_rng(SEED).choice(len(y), MAX_FIT_ROWS, replace=False)
        y, proba = y[rows], proba[rows]
    if proba.shape[1] == 2:
        return _fit_binary(metric, greater_is_better, y, proba)
    w = np.ones(proba.shape[1])
    best = _score(metric, y, proba, w)
    for _ in range(PASSES):
        improved = False
        for cls in range(len(w)):
            for g in GRID:
                trial = w.copy()
                trial[cls] = g
                score = _score(metric, y, proba, trial)
                if (score > best) if greater_is_better else (score < best):
                    best, w, improved = score, trial, True
        if not improved:
            break
    return w


def _fit_binary(metric: str, greater_is_better: bool, y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Лучшее отношение множителей (1, r) по сетке BINARY_RATIOS; при равенстве — ближайшее к 1 (argmax)."""
    best_w, best = np.ones(2), _score(metric, y, proba, np.ones(2))
    for r in sorted(BINARY_RATIOS, key=lambda r: abs(np.log(r))):
        w = np.array([1.0, r])
        score = _score(metric, y, proba, w)
        if (score > best) if greater_is_better else (score < best):
            best, best_w = score, w
    return best_w


def cross_fit_scores(metric: str, greater_is_better: bool, folds: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
    """folds — [(y_valid, proba_valid)] по фолдам. Скор фолда k — с множителями, подобранными на остальных фолдах."""
    scores = []
    for k, (y_k, proba_k) in enumerate(folds):
        rest = [f for i, f in enumerate(folds) if i != k]
        w = fit_weights(
            metric, greater_is_better, np.concatenate([f[0] for f in rest]), np.vstack([f[1] for f in rest])
        )
        scores.append(_score(metric, y_k, proba_k, w))
    return scores
