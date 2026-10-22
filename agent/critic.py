"""Критик гипотез по правилам: отсеивает гипотезы до того, как на них потрачено время. Владелец — agent.

Часть A-L3-2 (роли «генератор → критик»), которая не требует LLM:
  * пустые для задачи гипотезы — log_target на классификации, class_weight на регрессии (= бейзлайн);
  * недопустимые значения параметров — learning_rate ≤ 0, n_folds < 2 и т.п. (= гарантированный FAILED);
  * цена по времени — оценка длительности эксперимента относительно бейзлайна.

Модель времени откалибрована на multiclass (замер 2026-10-21):
learning_rate 0.05 → 0.02 дал ×1.6 (ранняя остановка, а не ×2.5), num_leaves 63 → 255 дал ×1.8,
шум самого времени от seed — до ×1.3, поэтому оценка берётся с запасом TIME_SAFETY.
"""

from __future__ import annotations

from interfaces.schemas import Hypothesis, TaskSpec, TaskType
from ml.experiment import BASE_PARAMS

DEFAULT_FOLDS = 5
TIME_SAFETY = 1.5
MAX_SLOWDOWN = 20.0  # оценка не больше ×20 — дальше модель времени всё равно не верна
# Во сколько раз модель дольше LightGBM при параметрах по умолчанию (замер ml, ревью #97; уточняется в #27)
MODEL_SLOWDOWN = {"lightgbm": 1.0, "catboost": 8.0, "xgboost": 4.0, "linear": 2.0}
UNKNOWN_MODEL_SLOWDOWN = 4.0


def review(hyp: Hypothesis, spec: TaskSpec) -> str | None:
    """Причина отклонить гипотезу или None, если по правилам она годится."""
    params = hyp.params
    regression = spec.task_type == TaskType.REGRESSION
    if params.get("log_target") and not regression:
        return "log_target действует только на регрессии — здесь гипотеза совпадёт с бейзлайном"
    if params.get("class_weight") is not None and regression:
        return "class_weight действует только на классификации — здесь гипотеза совпадёт с бейзлайном"
    if params.get("class_weight") not in (None, "balanced"):
        return f"class_weight={params['class_weight']!r}: допустимо только 'balanced'"
    n_folds = params.get("n_folds", DEFAULT_FOLDS)
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or not 2 <= n_folds <= 20:
        return f"n_folds={n_folds!r}: нужно целое от 2 до 20"
    model_params = params.get("model_params", {})
    lr = model_params.get("learning_rate", BASE_PARAMS["learning_rate"])
    if not isinstance(lr, int | float) or not 0 < lr <= 1:
        return f"learning_rate={lr!r}: нужно число в (0, 1]"
    model = _model(hyp)
    if model != "lightgbm":
        # CatBoost принимает num_leaves только при grow_policy=Lossguide, иначе параметр ничего не меняет
        if "num_leaves" in model_params and model_params.get("grow_policy") != "Lossguide":
            return f"num_leaves не действует в {model} — гипотеза совпадёт с моделью по умолчанию"
        return None
    leaves = model_params.get("num_leaves", BASE_PARAMS["num_leaves"])
    if not isinstance(leaves, int) or isinstance(leaves, bool) or not 2 <= leaves <= 4096:
        return f"num_leaves={leaves!r}: нужно целое от 2 до 4096"
    return None


def _model(hyp: Hypothesis) -> str:
    return str(hyp.params.get("model") or "lightgbm").lower()


def slowdown(hyp: Hypothesis) -> float:
    """Во сколько раз эксперимент дольше бейзлайна-LightGBM (без запаса). Не меньше 1: дешевле не считаем."""
    model_params = hyp.params.get("model_params", {})
    try:
        lr = float(model_params.get("learning_rate", BASE_PARAMS["learning_rate"]))
        leaves = float(model_params.get("num_leaves", BASE_PARAMS["num_leaves"]))
        folds = float(hyp.params.get("n_folds", DEFAULT_FOLDS))
    except (TypeError, ValueError):
        return 1.0
    model = _model(hyp)
    factor = MODEL_SLOWDOWN.get(model, UNKNOWN_MODEL_SLOWDOWN)
    if model == "lightgbm":  # learning_rate и num_leaves откалиброваны только для LightGBM
        factor *= lightgbm_slowdown(lr, leaves, BASE_PARAMS["learning_rate"], BASE_PARAMS["num_leaves"])
    factor *= max(1.0, folds / DEFAULT_FOLDS)
    return min(max(factor, 1.0), MAX_SLOWDOWN)


def lightgbm_slowdown(lr: float, leaves: float, base_lr: float, base_leaves: float) -> float:
    """Замедление LightGBM относительно бейзлайна с base_lr и base_leaves. Показатели степени — из замера
    2026-10-21 (бейзлайн тогда был 0.05 и 63); бейзлайн передаётся аргументом, поэтому смена BASE_PARAMS
    не ломает калибровку: оценка считается от того бейзлайна, время которого реально замерено в прогоне."""
    factor = 1.0
    if 0 < lr < base_lr:
        factor *= (base_lr / lr) ** 0.5
    if leaves > base_leaves:
        factor *= (leaves / base_leaves) ** 0.4
    return factor


def estimated_seconds(hyp: Hypothesis, baseline_seconds: float) -> float:
    """Оценка длительности эксперимента с запасом: ей сравниваем с оставшимся бюджетом."""
    seconds = baseline_seconds * slowdown(hyp) * TIME_SAFETY
    optuna = hyp.params.get("optuna")
    if optuna is True:
        optuna = {}  # optuna: true — подбор со значениями по умолчанию (ml/tuning.py: timeout 600 с)
    if isinstance(optuna, dict):  # подбор идёт до CV и длится до своего timeout
        try:
            seconds += float(optuna.get("timeout", 600))
        except (TypeError, ValueError):
            pass
    return seconds
