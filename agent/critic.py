"""Критик гипотез по правилам: отсеивает гипотезы до того, как на них потрачено время. Владелец — agent.

Часть A-L3-2 (роли «генератор → критик»), которая не требует LLM:
  * пустые для задачи гипотезы — log_target на классификации, class_weight на регрессии (= бейзлайн);
  * недопустимые значения параметров — learning_rate ≤ 0, n_folds < 2 и т.п. (= гарантированный FAILED);
  * цена по времени — оценка длительности эксперимента относительно бейзлайна;
  * потолок одной гипотезы (#276) — доля оставшегося времени, оценка поправляется по замеру прошлых запусков модели.

Модель времени откалибрована на multiclass (замер 2026-10-21):
learning_rate 0.05 → 0.02 дал ×1.6 (ранняя остановка, а не ×2.5), num_leaves 63 → 255 дал ×1.8,
шум самого времени от seed — до ×1.3, поэтому оценка берётся с запасом TIME_SAFETY.
"""

from __future__ import annotations

import os

from interfaces.schemas import ExperimentResult, ExperimentStatus, Hypothesis, RunBudget, TaskSpec, TaskType
from ml.experiment import BASE_PARAMS

DEFAULT_FOLDS = 5
TIME_SAFETY = 1.5
MAX_SLOWDOWN = 20.0  # оценка не больше ×20 — дальше модель времени всё равно не верна
# Во сколько раз модель дольше LightGBM при параметрах по умолчанию (замер ml, ревью #97; уточняется в #27)
MODEL_SLOWDOWN = {"lightgbm": 1.0, "catboost": 8.0, "xgboost": 4.0, "linear": 2.0}
UNKNOWN_MODEL_SLOWDOWN = 4.0
# Потолок одной гипотезы (#276): не больше этой доли оставшегося времени, чтобы одна дорогая модель не съела
# прогон (multiclass, 3fc774f: две гипотезы XGBoost — 60 мин из 88). 0 — без потолка и поправки, как раньше
HYP_TIME_SHARE_ENV = "AGENT_HYP_TIME_SHARE"
DEFAULT_HYP_TIME_SHARE = 0.25
HYP_FLOOR_BASELINES = 3.0  # потолок не ниже трёх бейзлайнов: варианты LightGBM (до ×2 с запасом) не режем


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


def hyp_time_share() -> float:
    """Доля оставшегося времени на одну гипотезу из AGENT_HYP_TIME_SHARE; мусор → по умолчанию, ≤ 0 → выключено."""
    try:
        share = float(os.getenv(HYP_TIME_SHARE_ENV, DEFAULT_HYP_TIME_SHARE))
    except ValueError:
        return DEFAULT_HYP_TIME_SHARE
    return min(share, 1.0) if share > 0 else 0.0


def hypothesis_limit(seconds_left: float, share: float, baseline_seconds: float) -> float:
    """Потолок времени одной гипотезы: доля остатка, но не меньше HYP_FLOOR_BASELINES бейзлайнов и не больше остатка."""
    return min(seconds_left, max(share * seconds_left, HYP_FLOOR_BASELINES * baseline_seconds))


def capped_budget(budget: RunBudget, limit: float) -> RunBudget:
    """Копия бюджета, в которой seconds_left() не больше limit: проверки ml/ внутри эксперимента (время на фолд,
    прогноз по первому фолду) останавливают гипотезу по потолку, а не по концу всего прогона."""
    cut = budget.seconds_left() - limit
    if cut <= 0:
        return budget
    return budget.model_copy(update={"total_seconds": budget.total_seconds - cut})


def observed_slowdown(
    hyp: Hypothesis, history: list[tuple[Hypothesis, ExperimentResult]], baseline_seconds: float
) -> float:
    """Поправка к оценке по замеру: во сколько раз прошлые запуски той же модели шли дольше, чем обещала оценка.

    Таблица MODEL_SLOWDOWN мерилась на одной задаче; на другой (больше классов, строк, категорий) XGBoost и CatBoost
    бывают дороже в разы. Берём худшее отношение «замер / оценка» среди завершённых и оборванных запусков модели
    (оборванный — пересчёт на все фолды по пройденным). Только вверх: дешевле оценки не считаем. Optuna и ансамбли
    не поправляем: у подбора свой лимит, а цена ансамбля складывается из разных моделей.
    """
    if baseline_seconds <= 0 or hyp.params.get("optuna") or hyp.params.get("ensemble"):
        return 1.0
    model = _model(hyp)
    worst = 1.0
    for past, result in history:
        if _model(past) != model or past.params.get("optuna") or past.params.get("ensemble"):
            continue
        if result.status not in (ExperimentStatus.OK, ExperimentStatus.TIMEOUT) or result.fit_seconds <= 0:
            continue
        seconds = result.fit_seconds
        if result.status == ExperimentStatus.TIMEOUT:
            if not result.fold_scores:
                continue
            try:
                folds = float(past.params.get("n_folds", DEFAULT_FOLDS))
            except (TypeError, ValueError):
                folds = DEFAULT_FOLDS
            seconds *= max(1.0, folds / len(result.fold_scores))
        worst = max(worst, seconds / (baseline_seconds * slowdown(past)))
    return worst
