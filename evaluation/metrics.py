"""Реестр метрик. Скрытые задачи могут прийти с ЛЮБОЙ метрикой — чем шире реестр, тем лучше.

Каждая метрика знает:
  * greater_is_better — направление;
  * prediction — что ей нужно на вход: "value" (число), "label" (метка класса), "proba" (вероятности);
  * aliases — как её могут назвать в task.md (ищем без учёта регистра, как отдельное слово).
    «*» в alias — любое окончание слова, для русских падежей: «средн* абсолютн* ошибк*» найдёт
    и «средняя абсолютная ошибка», и «по средней абсолютной ошибке».

На входе метрик классификации — то, что даёт ml/experiment.py: метки закодированы числами 0..k-1,
вероятности — матрица (n, k); «положительный» класс бинарной задачи — последний (код 1).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn import metrics as skm
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable
    greater_is_better: bool
    prediction: Literal["value", "label", "proba"]
    aliases: tuple[str, ...] = field(default_factory=tuple)


# --- регрессия


def _rmse(y, p):
    return float(np.sqrt(skm.mean_squared_error(y, p)))


def _rmsle(y, p):
    return float(np.sqrt(skm.mean_squared_log_error(y, np.clip(p, 0, None))))


def _msle(y, p):
    return float(skm.mean_squared_log_error(y, np.clip(p, 0, None)))


def _smape(y, p):
    """Симметричная MAPE в процентах (0–200), как на Kaggle; 0/0 считаем нулевой ошибкой."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    denom = np.abs(y) + np.abs(p)
    ratio = np.divide(2 * np.abs(p - y), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(100 * ratio.mean())


def _wape(y, p):
    """Weighted APE: сумма ошибок / сумма модулей факта (устойчива к нулям в y, в отличие от MAPE)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    total = np.abs(y).sum()
    return float(np.abs(y - p).sum() / total) if total else 0.0


# --- классификация по меткам


def _binary_or_macro(fn: Callable) -> Callable:
    """Бинарная метрика (F1, precision, recall); если классов больше двух — macro, а не ошибка sklearn."""

    def score(y, p):
        average = "binary" if len(np.union1d(np.unique(y), np.unique(p))) <= 2 else "macro"
        return float(fn(y, p, average=average, zero_division=0))

    return score


# --- классификация по вероятностям


def _positive_proba(proba):
    proba = np.asarray(proba)
    return proba[:, -1] if proba.ndim == 2 and proba.shape[1] == 2 else proba


def _roc_auc(y, proba, average="macro"):
    proba = _positive_proba(proba)
    if proba.ndim == 1:
        return float(skm.roc_auc_score(y, proba))
    return float(skm.roc_auc_score(y, proba, multi_class="ovr", average=average))


def _gini(y, proba):
    return 2 * _roc_auc(y, proba) - 1


def _pr_auc(y, proba):
    """Average precision (площадь под PR-кривой); для многих классов — macro по схеме «один против всех»."""
    proba = _positive_proba(proba)
    if proba.ndim == 1:
        return float(skm.average_precision_score(y, proba))
    return float(skm.average_precision_score(label_binarize(y, classes=range(proba.shape[1])), proba))


def _logloss(y, proba):
    return float(skm.log_loss(y, proba))


METRICS: dict[str, Metric] = {
    m.name: m
    for m in [
        # --- регрессия
        Metric(
            "rmse",
            _rmse,
            False,
            "value",
            ("rmse", "root mean squared error", "root mean square error", "корень из среднеквадратичн*"),
        ),
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
            ("mae", "mean absolute error", "средн* абсолютн* ошибк*", "средн* абсолютн* отклонени*"),
        ),
        Metric(
            "mape",
            lambda y, p: float(skm.mean_absolute_percentage_error(y, p)),
            False,
            "value",
            ("mape", "mean absolute percentage", "средн* абсолютн* процентн*"),
        ),
        Metric(
            "smape",
            _smape,
            False,
            "value",
            ("smape", "symmetric mean absolute percentage", "симметричн* mape", "симметричн* средн* абсолютн*"),
        ),
        Metric(
            "wape",
            _wape,
            False,
            "value",
            ("wape", "wmape", "weighted absolute percentage", "weighted mean absolute percentage"),
        ),
        Metric(
            "rmsle",
            _rmsle,
            False,
            "value",
            ("rmsle", "root mean squared log", "root mean squared logarithmic error", "root mean square log*"),
        ),
        Metric(
            "msle",
            _msle,
            False,
            "value",
            ("msle", "mean squared log", "mean squared logarithmic error"),
        ),
        Metric(
            "medae",
            lambda y, p: float(skm.median_absolute_error(y, p)),
            False,
            "value",
            ("medae", "median absolute error", "медианн* абсолютн* ошибк*"),
        ),
        Metric(
            "r2",
            lambda y, p: float(skm.r2_score(y, p)),
            True,
            "value",
            ("r2", "r^2", "r²", "r-squared", "r squared", "коэффициент* детерминации"),
        ),
        # --- классификация по меткам
        Metric(
            "f1_macro",
            lambda y, p: float(skm.f1_score(y, p, average="macro")),
            True,
            "label",
            (
                "macro f1",
                "f1 macro",
                "macro-f1",
                "f1_macro",
                "macro-averaged f1",
                "macro averaged f1",
                "f1-score (macro)",
                "f1 (macro)",
                "макро f1",
                "макро-f1",
                'average="macro"',
            ),
        ),
        Metric(
            "f1_weighted",
            lambda y, p: float(skm.f1_score(y, p, average="weighted")),
            True,
            "label",
            ("weighted f1", "f1 weighted", "f1_weighted", "weighted-f1", "взвешенн* f1"),
        ),
        Metric(
            "f1_micro",
            lambda y, p: float(skm.f1_score(y, p, average="micro")),
            True,
            "label",
            ("micro f1", "f1 micro", "f1_micro", "micro-f1"),
        ),
        Metric(
            "f1",
            _binary_or_macro(skm.f1_score),
            True,
            "label",
            ("f1-score", "f1 score", "f1-мера", "f1-мерой", "f-measure", "f1"),
        ),
        Metric(
            "precision",
            _binary_or_macro(skm.precision_score),
            True,
            "label",
            ("precision",),
        ),
        Metric(
            "recall",
            _binary_or_macro(skm.recall_score),
            True,
            "label",
            ("recall", "true positive rate"),  # «полнота», «sensitivity» — слишком частые слова в описании данных
        ),
        Metric(
            "accuracy",
            lambda y, p: float(skm.accuracy_score(y, p)),
            True,
            "label",
            ("accuracy", "доля правильных", "доля верных", "точность классификации"),
        ),
        Metric(
            "balanced_accuracy",
            lambda y, p: float(skm.balanced_accuracy_score(y, p)),
            True,
            "label",
            ("balanced accuracy", "balanced_accuracy", "сбалансированн* точност*"),
        ),
        Metric(
            "mcc",
            lambda y, p: float(skm.matthews_corrcoef(y, p)),
            True,
            "label",
            ("mcc", "matthews", "мэтьюс*"),
        ),
        Metric(
            "qwk",
            lambda y, p: float(skm.cohen_kappa_score(y, p, weights="quadratic")),
            True,
            "label",
            ("quadratic weighted kappa", "qwk", "cohen", "квадратичн* взвешенн* капп*"),
        ),
        # --- классификация по вероятностям
        Metric(
            "roc_auc",
            _roc_auc,
            True,
            "proba",
            (
                "roc-auc",
                "roc auc",
                "roc_auc",
                "auc-roc",
                "auc roc",
                "rocauc",
                "auc",
                "area under the roc curve",
                "area under the receiver operating characteristic",
                "площадь под roc",
            ),
        ),
        Metric(
            "roc_auc_weighted",
            lambda y, p: _roc_auc(y, p, average="weighted"),
            True,
            "proba",
            ("weighted roc-auc", "weighted roc auc", "weighted auc", "roc-auc (weighted)", "roc auc weighted"),
        ),
        Metric(
            "gini",
            _gini,
            True,
            "proba",
            ("gini", "normalized gini", "джини"),
        ),
        Metric(
            "pr_auc",
            _pr_auc,
            True,
            "proba",
            (
                "pr-auc",
                "pr auc",
                "pr_auc",
                "auc-pr",
                "auc pr",
                "average precision",
                "area under the precision-recall curve",
                "площадь под pr",
            ),
        ),
        Metric(
            "logloss",
            _logloss,
            False,
            "proba",
            (
                "logloss",
                "log loss",
                "log_loss",
                "log-loss",
                "binary cross-entropy",
                "binary crossentropy",
                "cross-entropy",
                "cross entropy",
                "crossentropy",
                "bce",
                "кросс-энтропи*",
            ),
        ),
    ]
}

# Порядок проверки при поиске в тексте: при совпадении позиции побеждает тот, кто раньше в списке.
# Сначала длинные/специфичные имена, иначе «f1» перебьёт «f1 macro», а «auc» — «auc-pr».
SEARCH_ORDER = [
    "f1_macro",
    "f1_weighted",
    "f1_micro",
    "balanced_accuracy",
    "rmsle",
    "msle",
    "rmse",
    "smape",
    "wape",
    "mape",
    "medae",
    "mae",
    "mse",
    "r2",
    "roc_auc_weighted",
    "pr_auc",
    "gini",
    "roc_auc",
    "logloss",
    "mcc",
    "qwk",
    "accuracy",
    "f1",
    "precision",
    "recall",
]


def compute_metric(name: str, y_true, y_pred) -> float:
    """Посчитать метрику по имени из реестра."""
    if name not in METRICS:
        raise KeyError(f"Неизвестная метрика «{name}». Есть: {sorted(METRICS)}")
    return METRICS[name].fn(y_true, y_pred)


# Фразы, после которых обычно называют основную метрику: ищем сначала рядом с ними (по порядку списка)
MARKERS = (
    "основная метрика",
    "основной метрикой",
    "метрика оценки",
    "evaluation metric",
    "scored with",
    "scored by",
    "scored using",
    "evaluated on",
    "evaluated using",
    "evaluated by",
    "evaluated with",
    "оценивается по",
    "оцениваются по",
    "метрика —",
    "метрика:",
    "metric",
)
# Отрицание ПЕРЕД метрикой относится к ней: «а не accuracy», «instead of accuracy», «в отличие от accuracy».
# Между отрицанием и метрикой — не больше 12 символов без знаков препинания (чтобы не задеть соседнюю фразу).
_NEGATION_BEFORE = re.compile(
    r"\b(?:not|unlike|instead of|rather than|except|не|вместо|в отличие от|кроме)\b[^.!?(),;:]{0,12}$"
)
# Отрицание ПОСЛЕ метрики: «accuracy is misleading», «accuracy здесь не подходит», «accuracy не используется».
_NEGATION_AFTER = re.compile(
    r"^[^.!?(),;:]{0,12}\b(?:is misleading|misleading|is not used|not used|do not use|don't use|for reference"
    r"|не используется|не используем|не подходит|не учитывается|для справки)\b"
)


def find_metric_in_text(text: str) -> str | None:
    """Найти основную метрику в тексте task.md. Возвращает ключ реестра или None.

    Сначала ищем рядом с фразами вроде «основная метрика» и «scored with», потом во всём тексте.
    Упоминания с отрицанием («we do NOT use accuracy», «Macro F1, а не Accuracy») пропускаем.
    """
    low = text.lower()
    for marker in MARKERS:
        pos = low.find(marker)
        if pos != -1:
            found = _first_metric(low, pos, pos + 200)
            if found:
                return found
    return _first_metric(low, 0, len(low))


def _first_metric(text: str, start: int, end: int) -> str | None:
    """Первая (по позиции) метрика без отрицания в text[start:end]; контекст отрицаний — весь text."""
    best: tuple[int, str] | None = None
    for name in SEARCH_ORDER:
        for alias in METRICS[name].aliases:
            for pos, stop in _find_words(text, alias, start, end):
                if best is not None and pos >= best[0]:
                    break
                if not _is_negated(text, pos, stop):
                    best = (pos, name)
                    break
    return best[1] if best else None


def _is_negated(text: str, pos: int, stop: int) -> bool:
    return bool(_NEGATION_BEFORE.search(text[max(0, pos - 40) : pos]) or _NEGATION_AFTER.search(text[stop : stop + 60]))


def _find_words(text: str, alias: str, start: int, end: int):
    """Все вхождения alias как отдельного слова в text[start:end] → (начало, конец) по порядку.

    Отдельное слово — чтобы «mse» не находилось внутри «rmse».
    «*» в alias — любое окончание слова: «взвешенн* f1» найдёт «взвешенная f1» и «взвешенной f1».
    """
    pattern = r"\w*".join(re.escape(part) for part in alias.split("*"))
    for match in re.compile(pattern).finditer(text, start, end):
        pos, stop = match.start(), match.end()
        before = text[pos - 1] if pos > 0 else " "
        after = text[stop] if stop < len(text) else " "
        if not before.isalnum() and (alias.endswith("*") or not after.isalnum()):
            yield pos, stop
