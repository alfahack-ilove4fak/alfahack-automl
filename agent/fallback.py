"""Запасные направления цикла (#200): что пробовать, когда playbook и LLM больше ничего не дают. Владелец — agent.

Без них агент выходил из цикла через 4–8 минут из 180: «Гипотезы закончились». Направления — по убыванию пользы,
все строятся от текущего лучшего (его предобработка сохраняется) и работают без LLM:
  1. другие модели из ml/ (xgboost, catboost, hist_gb) — с предобработкой лучшего;
  2. Optuna (скрытый ключ params.optuna, ml/tuning.py) на схеме валидации прогона, с лимитом по остатку бюджета;
  3. комбинации «почти прошедших»: лучше текущего лучшего на ≥ NEAR_MIN_WINS фолдах, но правило принятия
     их не взяло (t < 2) — накладываем на лучшего по одной и попарно (#87).
Порог принятия не меняется: каждая запасная гипотеза проходит тот же accept() с t ≥ 2.
id зависит от params, поэтому после нового лучшего направления строятся заново, а повторы истории отсекаются.
AGENT_FALLBACK=0 выключает запасные направления (цикл выходит, как до #200).
"""

from __future__ import annotations

import hashlib
import os
from itertools import combinations

from agent.acceptance import fold_gains
from agent.hypotheses import DEFAULT_MODEL, params_key
from interfaces.schemas import ExperimentResult, ExperimentStatus, Hypothesis, HypothesisKind, RunBudget, TaskSpec

FALLBACK_MODELS = ("xgboost", "catboost", "hist_gb")
NEAR_MIN_WINS = 3  # «почти прошла»: лучше лучшего хотя бы на 3 фолдах из 5
MAX_NEAR = 3  # сколько почти прошедших комбинируем: 3 по одной + 3 пары
OPTUNA_MAX_SECONDS = 600.0  # как DEFAULTS["timeout"] в ml/tuning.py
OPTUNA_BUDGET_SHARE = 0.3  # как BUDGET_SHARE в ml/tuning.py: подбор не съедает весь остаток
OPTUNA_MIN_SECONDS = 30.0  # меньше — подбор не успеет ни одного осмысленного испытания
OWN_KEYS = ("model", "model_params", "optuna")  # то, что запасная гипотеза задаёт сама
FALLBACK_ENV = "AGENT_FALLBACK"  # "0" — без запасных направлений
PREFIX = "fb-"  # id запасных гипотез


def is_fallback(hyp: Hypothesis) -> bool:
    return hyp.id.startswith(PREFIX)


def base_params(params: dict) -> dict:
    """params без служебных `_…` (подобраны под старый эксперимент) и validation (её добавит цикл)."""
    return {k: v for k, v in params.items() if not str(k).startswith("_") and k != "validation"}


def _key(params: dict) -> str:
    return params_key(base_params(params))


def _hyp(kind: HypothesisKind, title: str, rationale: str, params: dict, parent: Hypothesis) -> Hypothesis:
    digest = hashlib.sha1(_key(params).encode()).hexdigest()[:8]
    return Hypothesis(
        id=f"{PREFIX}{digest}",
        kind=kind,
        title=f"запасное: {title}",
        rationale=rationale,
        params=params,
        parent_id=parent.id,
    )


def _merge(base: dict, extra: dict) -> dict:
    """Наложить params почти прошедшей гипотезы на лучшего: model_params сливаются, feature_ops объединяются."""
    out = dict(base)
    for key, value in base_params(extra).items():
        if key == "model_params" and isinstance(value, dict):
            out[key] = {**(out.get(key) or {}), **value}
        elif key == "feature_ops" and isinstance(value, list):
            ops = list(out.get(key) or [])
            out[key] = ops + [op for op in value if op not in ops]
        else:
            out[key] = value
    return out


def _model(params: dict) -> str:
    return str(params.get("model") or DEFAULT_MODEL).lower()


def near_misses(
    history: list[tuple[Hypothesis, ExperimentResult]], best: tuple[Hypothesis, ExperimentResult], gib: bool
) -> list[Hypothesis]:
    """Гипотезы, которые лучше текущего лучшего по среднему и на ≥ NEAR_MIN_WINS фолдах, — от сильных к слабым."""
    found = []
    best_key = _key(best[0].params)
    for hyp, result in history:
        if result.status != ExperimentStatus.OK or _key(hyp.params) == best_key:
            continue
        if hyp.code or hyp.params.get("optuna"):
            continue  # код признаков и подобранные Optuna параметры через params не переносятся
        gains = fold_gains(result, best[1], gib)
        if gains and sum(g > 0 for g in gains) >= NEAR_MIN_WINS and sum(gains) > 0:
            found.append((sum(gains), hyp))
    found.sort(key=lambda item: -item[0])
    return [hyp for _, hyp in found[:MAX_NEAR]]


def fallback_hypotheses(
    spec: TaskSpec,
    history: list[tuple[Hypothesis, ExperimentResult]],
    best: tuple[Hypothesis, ExperimentResult],
    budget: RunBudget,
) -> list[Hypothesis]:
    """Запасные гипотезы по убыванию пользы, без повторов истории. Пусто — направления исчерпаны."""
    if os.getenv(FALLBACK_ENV, "1") == "0":
        return []
    gib = spec.metric.greater_is_better
    best_hyp = best[0]
    base = base_params(best_hyp.params)
    common = {k: v for k, v in base.items() if k not in OWN_KEYS}  # предобработка лучшего: признаки, веса, log
    out: list[Hypothesis] = []

    for model in FALLBACK_MODELS:
        if model != _model(base):
            out.append(
                _hyp(
                    HypothesisKind.MODEL,
                    f"{model} с предобработкой лучшего",
                    "другое семейство моделей: playbook перебирает только LightGBM (#200)",
                    {**common, "model": model},
                    best_hyp,
                )
            )

    timeout = min(OPTUNA_MAX_SECONDS, budget.seconds_left() * OPTUNA_BUDGET_SHARE)
    if timeout >= OPTUNA_MIN_SECONDS:
        for model in dict.fromkeys((_model(base), DEFAULT_MODEL)):  # лучшая модель, затем LightGBM
            params = {**common, "model": model, "optuna": {"timeout": round(timeout)}}
            if model == _model(base) and base.get("model_params"):
                params["model_params"] = base["model_params"]  # подбор стартует с того, что уже лучше
            out.append(
                _hyp(
                    HypothesisKind.HYPERPARAMS,
                    f"Optuna для {model} на схеме прогона",
                    "подбор гиперпараметров на той же схеме валидации, что и CV (ADR 0013)",
                    params,
                    best_hyp,
                )
            )

    near = near_misses(history, best, gib)
    for group in [(h,) for h in near] + list(combinations(near, 2)):
        if any(_model(h.params) != _model(base) for h in group):
            continue  # параметры чужой модели на лучшего не накладываем
        params = base
        for h in group:
            params = _merge(params, h.params)
        names = " + ".join(h.id for h in group)
        out.append(
            _hyp(
                HypothesisKind.HYPERPARAMS,
                f"лучший + {names}",
                "почти прошедшие (лучше на большинстве фолдов, t < 2) вместе могут дать значимый выигрыш (#87)",
                params,
                best_hyp,
            )
        )

    seen = {_key(h.params) for h, _ in history} | {_key(best_hyp.params)}
    unique = []
    for hyp in out:
        key = _key(hyp.params)
        if key not in seen:
            seen.add(key)
            unique.append(hyp)
    return unique
