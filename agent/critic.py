"""Критик гипотез по правилам: отсеивает гипотезы до того, как на них потрачено время. Владелец — agent.

Часть A-L3-2 (роли «генератор → критик»), которая не требует LLM:
  * пустые для задачи гипотезы — log_target на классификации, class_weight на регрессии (= бейзлайн);
  * недопустимые значения параметров — learning_rate ≤ 0, n_folds < 2 и т.п. (= гарантированный FAILED);
  * цель обучения не подходит таргету — Poisson/Tweedie при отрицательных значениях, Gamma при ≤ 0
    (LightGBM падает: «at least one target label is negative»; на public regression — 2 итерации впустую);
  * цена по времени — оценка длительности эксперимента относительно бейзлайна.

Модель времени откалибрована на multiclass (замер 2026-10-21):
learning_rate 0.05 → 0.02 дал ×1.6 (ранняя остановка, а не ×2.5), num_leaves 63 → 255 дал ×1.8,
шум самого времени от seed — до ×1.3, поэтому оценка берётся с запасом TIME_SAFETY.
"""

from __future__ import annotations

from interfaces.schemas import DataProfile, Hypothesis, TaskSpec, TaskType
from ml.experiment import BASE_PARAMS, OBJECTIVE_KEYS

DEFAULT_FOLDS = 5
TIME_SAFETY = 1.5
MAX_SLOWDOWN = 20.0  # оценка не больше ×20 — дальше модель времени всё равно не верна
# Во сколько раз модель дольше LightGBM при параметрах по умолчанию (замер ml, ревью #97; уточняется в #27)
MODEL_SLOWDOWN = {"lightgbm": 1.0, "catboost": 8.0, "xgboost": 4.0, "linear": 2.0}
UNKNOWN_MODEL_SLOWDOWN = 4.0
NON_NEGATIVE_OBJECTIVES = ("poisson", "tweedie")  # подстрока имени цели: poisson, count:poisson, reg:tweedie, Tweedie:…
POSITIVE_OBJECTIVES = ("gamma",)  # gamma, reg:gamma — таргет строго > 0


def review(hyp: Hypothesis, spec: TaskSpec, profile: DataProfile | None = None) -> str | None:
    """Причина отклонить гипотезу или None, если по правилам она годится. profile — для проверки цели по таргету."""
    params = hyp.params
    regression = spec.task_type == TaskType.REGRESSION
    if regression and profile is not None and (reason := _objective_mismatch(hyp, profile)):
        return reason
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


def _objectives(hyp: Hypothesis) -> list[str]:
    """Явно заданные цели обучения гипотезы (model_params и участники ансамбля), в нижнем регистре."""
    blocks = [hyp.params.get("model_params")]
    members = hyp.params.get("ensemble")
    if isinstance(members, list):
        blocks += [m.get("model_params") for m in members if isinstance(m, dict)]
    return [str(b[k]).lower() for b in blocks if isinstance(b, dict) for k in OBJECTIVE_KEYS if b.get(k) is not None]


def _objective_mismatch(hyp: Hypothesis, profile: DataProfile) -> str | None:
    """Цель обучения, которую таргет не допускает: модель упадёт на первом фолде."""
    if hyp.params.get("log_target"):
        return None  # учимся на log1p(max(y, 0)) ≥ 0 — любые из этих целей допустимы
    low = profile.target_summary.get("min")
    if not isinstance(low, int | float):
        return None
    for objective in _objectives(hyp):
        if low < 0 and any(name in objective for name in NON_NEGATIVE_OBJECTIVES):
            return f"цель {objective} требует таргет ≥ 0, а минимум таргета {low:g} — модель упадёт"
        if low <= 0 and any(name in objective for name in POSITIVE_OBJECTIVES):
            return f"цель {objective} требует таргет > 0, а минимум таргета {low:g} — модель упадёт"
    return None


def _model(hyp: Hypothesis) -> str:
    return str(hyp.params.get("model") or "lightgbm").lower()


def slowdown(hyp: Hypothesis) -> float:
    """Во сколько раз эксперимент дольше бейзлайна-LightGBM (без запаса). Не меньше 1: дешевле не считаем.

    Ансамбль (params.ensemble) обучает каждого участника отдельно — его цена складывается из цен участников.
    """
    try:
        folds = float(hyp.params.get("n_folds", DEFAULT_FOLDS))
    except (TypeError, ValueError):
        return 1.0
    members = hyp.params.get("ensemble")
    if isinstance(members, list) and members:  # предел ×MAX_SLOWDOWN — у каждого участника, не у суммы
        factor = sum(min(_model_slowdown(*_ensemble_member(m)), MAX_SLOWDOWN) for m in members)
        limit = MAX_SLOWDOWN * len(members)
    else:
        factor = _model_slowdown(_model(hyp), hyp.params.get("model_params", {}))
        limit = MAX_SLOWDOWN
    factor *= max(1.0, folds / DEFAULT_FOLDS)
    return min(max(factor, 1.0), limit)


def _ensemble_member(member) -> tuple[str, dict]:
    if isinstance(member, dict):
        return str(member.get("model") or "lightgbm").lower(), member.get("model_params") or {}
    return str(member).lower(), {}


def _model_slowdown(model: str, model_params) -> float:
    """Замедление одной модели относительно бейзлайна-LightGBM."""
    factor = MODEL_SLOWDOWN.get(model, UNKNOWN_MODEL_SLOWDOWN)
    if model == "lightgbm" and isinstance(model_params, dict):  # lr и num_leaves откалиброваны только для LightGBM
        try:
            lr = float(model_params.get("learning_rate", BASE_PARAMS["learning_rate"]))
            leaves = float(model_params.get("num_leaves", BASE_PARAMS["num_leaves"]))
        except (TypeError, ValueError):
            return factor
        factor *= lightgbm_slowdown(lr, leaves, BASE_PARAMS["learning_rate"], BASE_PARAMS["num_leaves"])
    return factor


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
