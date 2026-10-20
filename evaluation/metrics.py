"""Реестр метрик. Скрытые задачи могут прийти с ЛЮБОЙ метрикой — чем шире реестр, тем лучше.

Каждая метрика знает:
  * greater_is_better — направление;
  * prediction — что ей нужно на вход: "value" (число), "label" (метка класса), "proba" (вероятности);
  * aliases — как её могут назвать в task.md (ищем без учёта регистра).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable
    greater_is_better: bool
    prediction: Literal["value", "label", "proba"]
    aliases: tuple[str, ...] = field(default_factory=tuple)


def _rmse(y, p):
    return float(np.sqrt(skm.mean_squared_error(y, p)))


def _rmsle(y, p):
    return float(np.sqrt(skm.mean_squared_log_error(y, np.clip(p, 0, None))))


def _roc_auc(y, proba):
    proba = np.asarray(proba)
    if proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1]
    if proba.ndim == 1:
        return float(skm.roc_auc_score(y, proba))
    return float(skm.roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def _logloss(y, proba):
    return float(skm.log_loss(y, proba))


METRICS: dict[str, Metric] = {
    m.name: m
    for m in [
        # --- регрессия
        Metric("rmse", _rmse, False, "value", ("rmse", "root mean squared error", "корень из среднеквадратичной")),
        Metric(
            "mse",
            lambda y, p: float(skm.mean_squared_error(y, p)),
            False,
            "value",
            ("mse", "mean squared error", "среднеквадратичн"),
        ),
        Metric(
            "mae",
            lambda y, p: float(skm.mean_absolute_error(y, p)),
            False,
            "value",
            ("mae", "mean absolute error", "средняя абсолютная"),
        ),
        Metric(
            "mape",
            lambda y, p: float(skm.mean_absolute_percentage_error(y, p)),
            False,
            "value",
            ("mape", "mean absolute percentage"),
        ),
        Metric("rmsle", _rmsle, False, "value", ("rmsle", "root mean squared log")),
        Metric(
            "medae",
            lambda y, p: float(skm.median_absolute_error(y, p)),
            False,
            "value",
            ("medae", "median absolute error"),
        ),
        Metric(
            "r2", lambda y, p: float(skm.r2_score(y, p)), True, "value", ("r2", "r^2", "r²", "коэффициент детерминации")
        ),
        # --- классификация по меткам
        Metric(
            "f1_macro",
            lambda y, p: float(skm.f1_score(y, p, average="macro")),
            True,
            "label",
            ("macro f1", "f1 macro", "macro-f1", "f1_macro", 'average="macro"'),
        ),
        Metric(
            "f1_weighted",
            lambda y, p: float(skm.f1_score(y, p, average="weighted")),
            True,
            "label",
            ("weighted f1", "f1 weighted", "f1_weighted"),
        ),
        Metric(
            "f1_micro",
            lambda y, p: float(skm.f1_score(y, p, average="micro")),
            True,
            "label",
            ("micro f1", "f1 micro", "f1_micro"),
        ),
        Metric("f1", lambda y, p: float(skm.f1_score(y, p)), True, "label", ("f1-score", "f1 score", "f1-мера")),
        Metric(
            "accuracy",
            lambda y, p: float(skm.accuracy_score(y, p)),
            True,
            "label",
            ("accuracy", "доля правильных", "точность классификации"),
        ),
        Metric(
            "balanced_accuracy",
            lambda y, p: float(skm.balanced_accuracy_score(y, p)),
            True,
            "label",
            ("balanced accuracy", "balanced_accuracy"),
        ),
        Metric("mcc", lambda y, p: float(skm.matthews_corrcoef(y, p)), True, "label", ("mcc", "matthews")),
        Metric(
            "qwk",
            lambda y, p: float(skm.cohen_kappa_score(y, p, weights="quadratic")),
            True,
            "label",
            ("quadratic weighted kappa", "qwk", "cohen"),
        ),
        # --- классификация по вероятностям
        Metric("roc_auc", _roc_auc, True, "proba", ("roc-auc", "roc auc", "roc_auc", "auc")),
        Metric(
            "logloss", _logloss, False, "proba", ("logloss", "log loss", "log_loss", "cross-entropy", "кросс-энтропи")
        ),
    ]
}

# Порядок проверки при поиске в тексте: сначала длинные/специфичные имена,
# иначе «f1» найдётся раньше «macro f1», а «mse» — внутри «rmse».
SEARCH_ORDER = [
    "f1_macro",
    "f1_weighted",
    "f1_micro",
    "balanced_accuracy",
    "rmsle",
    "rmse",
    "mape",
    "medae",
    "mae",
    "mse",
    "r2",
    "roc_auc",
    "logloss",
    "mcc",
    "qwk",
    "accuracy",
    "f1",
]


def compute_metric(name: str, y_true, y_pred) -> float:
    """Посчитать метрику по имени из реестра."""
    if name not in METRICS:
        raise KeyError(f"Неизвестная метрика «{name}». Есть: {sorted(METRICS)}")
    return METRICS[name].fn(y_true, y_pred)


def find_metric_in_text(text: str) -> str | None:
    """Найти основную метрику в тексте task.md. Возвращает ключ реестра или None."""
    low = text.lower()
    # Если есть явная фраза «основная метрика … X» — ищем сначала рядом с ней
    for marker in ("основная метрика", "метрика оценки", "метрика —", "метрика:", "metric"):
        pos = low.find(marker)
        if pos != -1:
            found = _first_metric(low[pos : pos + 200])
            if found:
                return found
    return _first_metric(low)


def _first_metric(text: str) -> str | None:
    best: tuple[int, str] | None = None
    for name in SEARCH_ORDER:
        for alias in METRICS[name].aliases:
            pos = _find_word(text, alias)
            if pos != -1 and (best is None or pos < best[0]):
                best = (pos, name)
    return best[1] if best else None


def _find_word(text: str, alias: str) -> int:
    """Поиск alias как отдельного слова (чтобы «mse» не находилось внутри «rmse»)."""
    start = 0
    while True:
        pos = text.find(alias, start)
        if pos == -1:
            return -1
        before = text[pos - 1] if pos > 0 else " "
        after = text[pos + len(alias)] if pos + len(alias) < len(text) else " "
        if not before.isalnum() and not after.isalnum():
            return pos
        start = pos + 1
