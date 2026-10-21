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

GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 17))  # ×0.25 … ×4, в середине ×1
PASSES = 3


def _score(metric: str, y: np.ndarray, proba: np.ndarray, w: np.ndarray) -> float:
    return compute_metric(metric, y, (proba * w).argmax(axis=1))


def fit_weights(metric: str, greater_is_better: bool, y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Покоординатный подъём по сетке GRID: по очереди для каждого класса — лучший множитель при остальных фиксированных."""
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
