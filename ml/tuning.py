"""Подбор гиперпараметров Optuna для params.optuna. Владелец — ML-pipeline инженер.

params.optuna: {trials: 30, timeout: 600, folds: 3, rows: 100000} — любые ключи можно опустить.
Поиск дешёвый: на подвыборке (rows) и меньшем числе фолдов (folds). Найденные model_params run_experiment подставляет
в гипотезу и затем делает обычную CV — её скор сравним с остальными гипотезами.

Таймаут жёсткий с точностью до одного фолда: после каждого фолда проверяется дедлайн, и испытание обрывается.
На подбор уходит не больше BUDGET_SHARE оставшегося времени агента, даже если timeout больше.
Слабые испытания отсекаются после первых фолдов (MedianPruner).
"""

from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd

from interfaces.schemas import Hypothesis, RunBudget, TaskSpec
from runtime.log import get_logger

log = get_logger("tuning")

DEFAULTS = {"trials": 30, "timeout": 600, "folds": 3, "rows": 100_000}
BUDGET_SHARE = 0.3


def _space(model: str, trial: optuna.Trial, classification: bool) -> dict:
    """Пространство поиска model_params для каждой модели реестра."""
    if model == "linear":  # шага обучения нет — подбираем только силу регуляризации
        key = "C" if classification else "alpha"
        return {key: trial.suggest_float(key, 1e-3, 100.0, log=True)}
    lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    if model == "lightgbm":
        return {
            "learning_rate": lr,
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    if model == "xgboost":
        return {
            "learning_rate": lr,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    if model == "catboost":
        return {
            "learning_rate": lr,
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        }
    if model == "hist_gb":
        return {
            "learning_rate": lr,
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        }
    raise ValueError(f"для модели {model!r} нет пространства поиска Optuna")


def tune(
    spec: TaskSpec,
    hypothesis: Hypothesis,
    x: pd.DataFrame,
    y: np.ndarray,
    y_fit: np.ndarray,
    ops: list[str],
    log_target: bool,
    budget: RunBudget,
) -> dict:
    """Лучшие model_params или {} (ни одно испытание не завершилось — остаются параметры гипотезы)."""
    from ml.experiment import DEFAULT_MODEL, SEED, _fold_score, _is_classification, _sample_rows, _split

    cfg = {**DEFAULTS, **(hypothesis.params.get("optuna") or {})}
    model = str(hypothesis.params.get("model") or DEFAULT_MODEL).strip().lower()
    classification = _is_classification(spec)
    timeout = max(0.0, min(float(cfg["timeout"]), budget.seconds_left() * BUDGET_SHARE))
    deadline = time.time() + timeout
    if len(y) > int(cfg["rows"]):
        rows = _sample_rows(spec, y, int(cfg["rows"]))
        x, y, y_fit = x.iloc[rows], y[rows], y_fit[rows]
    folds = _split(spec, y, int(cfg["folds"]))
    base = dict(hypothesis.params.get("model_params") or {})

    def objective(trial: optuna.Trial) -> float:
        trial_params = {**hypothesis.params, "model_params": {**base, **_space(model, trial, classification)}}
        hyp = hypothesis.model_copy(update={"params": trial_params})
        scores = []
        for step, (tr, va) in enumerate(folds):
            if time.time() > deadline:
                raise optuna.TrialPruned()
            score, _ = _fold_score(spec, hyp, x, y, y_fit, tr, va, ops, log_target)
            scores.append(score)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize" if spec.metric.greater_is_better else "minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    started = time.time()
    study.optimize(objective, n_trials=int(cfg["trials"]), timeout=timeout, catch=(Exception,))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        log.warning("Optuna: ни одно испытание не завершилось за %.0f с — остаются параметры гипотезы", timeout)
        return {}
    best = dict(study.best_params)
    log.info(
        "Optuna (%s): %d испытаний (%d завершено) за %.0f с, лучший %s=%.5f, параметры %s",
        model,
        len(study.trials),
        len(done),
        time.time() - started,
        spec.metric.name,
        study.best_value,
        best,
    )
    return best
