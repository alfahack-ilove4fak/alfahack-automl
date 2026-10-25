"""Запасные направления цикла (#200): что пробовать, когда playbook и LLM больше ничего не дают. Владелец — agent.

Без них агент выходил из цикла через 4–8 минут из 180: «Гипотезы закончились». Набор и порядок — список danil
(стенд на 10 задачах, комментарий в #200), дальше — направления без предела по числу, чтобы время не простаивало.
Все строятся от текущего лучшего (его предобработка и код признаков сохраняются) и работают без LLM:
  1. другие модели и ансамбли: xgboost → ансамбль [лучшая, LightGBM/XGBoost] → catboost → Optuna лучшей модели →
     ансамбль из трёх → hist_gb → Optuna LightGBM;
  2. комбинации «почти прошедших»: лучше текущего лучшего на ≥ NEAR_MIN_WINS фолдах, но правило принятия
     их не взяло (t < 2) — накладываем на лучшего по одной и попарно (#87);
  3. playbook поверх лучшего: гипотезы playbook проверялись против бейзлайна по отдельности, а после нового
     лучшего (log_target, другая модель) те же идеи вместе с ним — уже другие гипотезы (#87);
  4. усреднение seed лучшей модели (#32): ансамбли из SEED_BAGS копий с разным random_state — меньше дисперсия;
  5. Optuna других моделей и подбор глубже (больше испытаний того же TPE с тем же seed) — дорого, но полезно,
     пока есть время.
Порог принятия не меняется: каждая запасная гипотеза проходит тот же accept() с t ≥ 2.
id зависит от params, поэтому после нового лучшего направления строятся заново, а повторы истории отсекаются.
AGENT_FALLBACK=0 выключает запасные направления (цикл выходит, как до #200).
"""

from __future__ import annotations

import hashlib
import os
from itertools import combinations

from agent.acceptance import fold_gains
from agent.hypotheses import DEFAULT_MODEL, _applies, load_playbook, params_key, run_scheme, with_task_defaults
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    RunBudget,
    TaskSpec,
)

FALLBACK_MODELS = ("xgboost", "catboost", "hist_gb")
NEAR_MIN_WINS = 3  # «почти прошла»: лучше лучшего хотя бы на 3 фолдах из 5
MAX_NEAR = 3  # сколько почти прошедших комбинируем: 3 по одной + 3 пары
OPTUNA_MAX_SECONDS = 600.0  # как DEFAULTS["timeout"] в ml/tuning.py
OPTUNA_BUDGET_SHARE = 0.3  # как BUDGET_SHARE в ml/tuning.py: подбор не съедает весь остаток
OPTUNA_MIN_SECONDS = 30.0  # меньше — подбор не успеет ни одного осмысленного испытания
# схемы, где Optuna переобучается под фолды: на public/regression (time:) честно 53.1 против 44.5 без неё (#236)
NO_OPTUNA_SCHEMES = ("time:", "group:")
OPTUNA_TRIALS = 30  # как DEFAULTS["trials"] в ml/tuning.py
OPTUNA_DEEPER = (2, 4, 8)  # подбор глубже: ×2, ×4, ×8 испытаний — TPE с тем же seed продолжает тот же поиск
# усреднение seed: 3 → 5 → 8 → 13 копий лучшей модели. Принималось на гейте (leak_column_auc, mape_in_words) и,
# в отличие от нового раунда Optuna, не подгоняется под шум CV: победитель Optuna бывает удачей одного seed
# (cp1251_r2, extra_trees: r2 0.840 при seed 42, 0.777 при seed 43)
SEED_BAGS = (3, 5, 8, 13)
# Ключ seed у моделей реестра ml/; у linear и hist_gb без подвыборки seed почти ничего не меняет — их не усредняем
SEED_KEYS = {"lightgbm": "random_state", "xgboost": "random_state", "catboost": "random_seed"}
SEED = 42  # как ml.experiment.SEED: первая копия ансамбля — та же модель, что и лучшая
# то, что запасная гипотеза задаёт сама; остальное (признаки, веса, log_target, objective) — от лучшего
OWN_KEYS = ("model", "model_params", "optuna", "ensemble")
FALLBACK_ENV = "AGENT_FALLBACK"  # "0" — без запасных направлений
PREFIX = "fb-"  # id запасных гипотез


def is_fallback(hyp: Hypothesis) -> bool:
    return hyp.id.startswith(PREFIX)


def base_params(params: dict) -> dict:
    """params без служебных `_…` (подобраны под старый эксперимент) и validation (её добавит цикл)."""
    return {k: v for k, v in params.items() if not str(k).startswith("_") and k != "validation"}


def _key(params: dict) -> str:
    """Ключ повтора. Optuna — одно направление на модель, предобработку и глубину подбора: её timeout зависит от
    остатка бюджета, а model_params после подбора — от результата, поэтому их не учитываем (иначе на diabetes
    ≈ 70 повторов, #230). Глубина (trials) — учитываем: подбор глубже — это другой, более дорогой поиск."""
    params = base_params(params)
    if params.get("optuna"):
        optuna = params["optuna"]
        trials = optuna.get("trials") if isinstance(optuna, dict) else None
        params = {k: v for k, v in params.items() if k != "model_params"}
        params["optuna"] = {"trials": trials} if trials and trials != OPTUNA_TRIALS else True
    return params_key(params)


def _hyp(kind: HypothesisKind, title: str, rationale: str, params: dict, parent: Hypothesis) -> Hypothesis:
    digest = hashlib.sha1(_key(params).encode()).hexdigest()[:8]
    return Hypothesis(
        id=f"{PREFIX}{digest}",
        kind=kind,
        title=f"запасное: {title}",
        rationale=rationale,
        params=params,
        code=parent.code,  # код признаков лучшего (LLM, A-L3-1) — часть его предобработки
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


def _member(model: str, base: dict, seed: int | None = None) -> str | dict:
    """Участник ансамбля: модель лучшего — с её model_params (и своим seed), остальные — по умолчанию."""
    model_params = dict(base.get("model_params") or {}) if model == _model(base) else {}
    if seed is not None:
        model_params[SEED_KEYS[model]] = seed
    return {"model": model, "model_params": model_params} if model_params else model


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


def _fits(item: dict, spec: TaskSpec, profile: DataProfile | None) -> bool:
    """Гипотеза playbook применима к задаче; без профиля проверяем только тип задачи."""
    if profile is not None:
        return _applies(item, spec, profile)
    types = item.get("task_types")
    return not types or spec.task_type.value in types


class _Directions:
    """Сборщик запасных гипотез от одного лучшего: общие params и помощники для каждого направления."""

    def __init__(self, best_hyp: Hypothesis, budget: RunBudget, scheme: str | None = None):
        self.best = best_hyp
        self.scheme = scheme or ""
        self.base = base_params(best_hyp.params)
        # лучший без optuna: его подобранные model_params уже в base — подбирать заново при каждом наложении незачем
        self.tuned = {k: v for k, v in self.base.items() if k != "optuna"}
        self.common = {k: v for k, v in self.base.items() if k not in OWN_KEYS}  # предобработка лучшего
        self.model = _model(self.base)
        self.single = not self.base.get("ensemble")  # лучший — одна модель (не ансамбль)
        self.timeout = min(OPTUNA_MAX_SECONDS, budget.seconds_left() * OPTUNA_BUDGET_SHARE)
        self.out: list[Hypothesis] = []

    def add(self, kind: HypothesisKind, title: str, rationale: str, params: dict) -> None:
        self.out.append(_hyp(kind, title, rationale, params, self.best))

    def other_model(self, model: str) -> None:
        if model != self.model or not self.single:
            self.add(
                HypothesisKind.MODEL,
                f"{model} с предобработкой лучшего",
                "другое семейство моделей с тем, что уже помогло лучшему (#200)",
                {**self.common, "model": model},
            )

    def ensemble(self, size: int) -> None:
        if not self.single:
            return
        models = list(dict.fromkeys((self.model, "lightgbm", "xgboost", "catboost")))[:size]
        self.add(
            HypothesisKind.ENSEMBLE,
            f"ансамбль {' + '.join(models)}",
            "среднее прогнозов разных семейств: ошибки моделей частично гасят друг друга (стенд #200)",
            {**self.common, "ensemble": [_member(m, self.base) for m in models]},
        )

    def optuna(self, model: str, depth: int = 1) -> None:
        if self.timeout < OPTUNA_MIN_SECONDS or (model == self.model and not self.single):
            return
        if self.scheme.startswith(NO_OPTUNA_SCHEMES):
            return  # подбор на таких схемах хуже бейзлайна
        cfg: dict = {"timeout": round(self.timeout)}
        if depth > 1:
            cfg["trials"] = OPTUNA_TRIALS * depth
        params = {**self.common, "model": model, "optuna": cfg}
        if model == self.model and self.base.get("model_params"):
            params["model_params"] = self.base["model_params"]  # подбор стартует с того, что уже лучше
        title = f"Optuna для {model} на схеме прогона" + (f", {cfg['trials']} испытаний" if depth > 1 else "")
        self.add(
            HypothesisKind.HYPERPARAMS,
            title,
            "подбор гиперпараметров на той же схеме валидации, что и CV (ADR 0013)",
            params,
        )

    def seeds(self, n: int) -> None:
        if not self.single or self.model not in SEED_KEYS:
            return
        self.add(
            HypothesisKind.ENSEMBLE,
            f"усреднение {n} seed {self.model}",
            "та же модель с разными seed: подвыборки строк и признаков разные, среднее снижает дисперсию (#32)",
            {**self.common, "ensemble": [_member(self.model, self.base, SEED + i) for i in range(n)]},
        )

    def combo(self, group: tuple[Hypothesis, ...]) -> None:
        if not self.single or any(_model(h.params) != self.model for h in group):
            return  # параметры чужой модели на лучшего не накладываем
        params = self.tuned
        for h in group:
            params = _merge(params, h.params)
        self.add(
            HypothesisKind.HYPERPARAMS,
            f"лучший + {' + '.join(h.id for h in group)}",
            "почти прошедшие (лучше на большинстве фолдов, t < 2) вместе могут дать значимый выигрыш (#87)",
            params,
        )

    def playbook_on_best(self, spec: TaskSpec, profile: DataProfile | None) -> None:
        """Гипотезы playbook поверх лучшего: своя модель — слияние с его params, чужая — только его предобработка."""
        for item in load_playbook():
            params = item.get("params") or {}
            if "validation" in params or not _fits(item, spec, profile):
                continue
            if self.single and _model(params) == self.model:
                merged = _merge(self.tuned, params)
            else:
                merged = _merge(self.common, params)
            self.add(
                HypothesisKind(item["kind"]),
                f"{item['id']} поверх лучшего",
                "идея playbook проверялась против бейзлайна, а вместе с тем, что уже принято, — нет (#87)",
                with_task_defaults(merged, spec),
            )


def fallback_hypotheses(
    spec: TaskSpec,
    history: list[tuple[Hypothesis, ExperimentResult]],
    best: tuple[Hypothesis, ExperimentResult],
    budget: RunBudget,
    profile: DataProfile | None = None,
) -> list[Hypothesis]:
    """Запасные гипотезы по убыванию пользы, без повторов истории. Пусто — направления исчерпаны.

    profile нужен только для «playbook поверх лучшего» (какие гипотезы применимы к задаче); без него — все.
    """
    if os.getenv(FALLBACK_ENV, "1") == "0":
        return []
    d = _Directions(best[0], budget, run_scheme(history))
    # 1. модели и ансамбли — порядок danil по стенду (#200): дешёвые и сильные раньше
    d.other_model("xgboost")
    d.ensemble(2)
    d.other_model("catboost")
    d.optuna(d.model)
    d.ensemble(3)
    d.other_model("hist_gb")
    d.optuna(DEFAULT_MODEL)
    # 2. почти прошедшие (#87)
    near = near_misses(history, best, spec.metric.greater_is_better)
    for group in [(h,) for h in near] + list(combinations(near, 2)):
        d.combo(group)
    # 3–5. без предела по числу идей: время на платформе не должно простаивать
    d.playbook_on_best(spec, profile)
    for n in SEED_BAGS:
        d.seeds(n)
    for model in ("xgboost", "catboost"):
        d.optuna(model)
    for depth in OPTUNA_DEEPER:
        d.optuna(d.model, depth)

    seen = {_key(h.params) for h, _ in history} | {_key(best[0].params)}
    unique = []
    for hyp in d.out:
        key = _key(hyp.params)
        if key not in seen:
            seen.add(key)
            unique.append(hyp)
    return unique
