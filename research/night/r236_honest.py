"""#236: честная проверка регрессии — какие гипотезы и какая финальная модель честно лучше бейзлайна.

    uv run python research/night/r236_honest.py [--only baseline,optuna_time] [--optuna-seconds 120]

Train — годы 1–4 public/regression, честная отложенная часть — год 5 (как в харнессе: time:Year_Factor>=5).
Для каждой гипотезы: CV в схеме, которую выбрал бы агент (choose_validation), и в KFold; затем финальная модель
(fit_predict_final) на годах 1–4 → RMSE на годе 5. Варианты числа деревьев финала: как в агенте (поправка #218
на долю train во фолдах time-схемы) и без поправки.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.hypotheses import baseline_hypothesis  # noqa: E402
from agent.task_parser import parse_task  # noqa: E402
from interfaces.schemas import Hypothesis, HypothesisKind  # noqa: E402
from ml.experiment import fit_predict_final, run_experiment  # noqa: E402
from ml.profiling import load_data, profile_data  # noqa: E402
from ml.validation import choose_validation  # noqa: E402
from runtime.budget import make_budget  # noqa: E402


def hyp(hid: str, params: dict) -> Hypothesis:
    return Hypothesis(id=hid, kind=HypothesisKind.HYPERPARAMS, title=hid, rationale="#236", params=params)


def candidates(optuna_seconds: int) -> dict[str, dict]:
    base = baseline_hypothesis().params
    return {
        "baseline": dict(base),
        "lgbm_default": {
            "model": "lightgbm",
            "model_params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
        },
        "linear": {"model": "linear"},
        "xgboost": {"model": "xgboost"},
        "catboost": {"model": "catboost"},
        "extra_trees": {"model": "lightgbm", "model_params": {"extra_trees": True}},
        "slow_regularized": {
            "model": "lightgbm",
            "model_params": {
                "learning_rate": 0.02,
                "num_leaves": 31,
                "min_child_samples": 50,
                "reg_lambda": 1.0,
                "colsample_bytree": 0.5,
            },
        },
        "log_target": {"model": "lightgbm", "log_target": True},
        "optuna": {"model": "lightgbm", "optuna": {"timeout": optuna_seconds}},
        "ens_lgb_xgb": {"ensemble": ["lightgbm", "xgboost"]},
        "ens_lgb_xgb_cat": {"ensemble": ["lightgbm", "xgboost", "catboost"]},
        "ens_lgb_xgb_et": {"ensemble": ["lightgbm", "xgboost",
                                        {"model": "lightgbm", "model_params": {"extra_trees": True}}]},
        "ens_4": {"ensemble": ["lightgbm", "xgboost", {"model": "lightgbm", "model_params": {"extra_trees": True}},
                               {"model": "lightgbm", "model_params": {"learning_rate": 0.02, "num_leaves": 31,
                                "min_child_samples": 50, "reg_lambda": 1.0, "colsample_bytree": 0.5}}]},
        "ens_lgb_linear": {"ensemble": ["lightgbm", "linear"]},
    }


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--schemes", default="agent,kfold")
    ap.add_argument("--optuna-seconds", type=int, default=120)
    ap.add_argument("--out", default="runs/236/honest.jsonl")
    args = ap.parse_args()
    spec = parse_task(ROOT / "tasks/public/regression/task.md")
    full, _ = load_data(spec)
    train = full[full["Year_Factor"] <= 4].reset_index(drop=True)
    hold = full[full["Year_Factor"] == 5].reset_index(drop=True)
    y_hold = hold[spec.target].to_numpy()
    test = hold.drop(columns=[spec.target])
    profile = profile_data(spec, train, test)
    agent_scheme = choose_validation(spec, profile, train, test)
    print("схема агента:", agent_scheme, flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, params in candidates(args.optuna_seconds).items():
        if args.only and name not in args.only.split(","):
            continue
        for scheme_name in args.schemes.split(","):
            scheme = agent_scheme if scheme_name == "agent" else "kfold"
            h = hyp(name, {**copy.deepcopy(params), "validation": scheme})
            t = time.time()
            res = run_experiment(spec, profile, h, train, make_budget(time.time()))
            row = {
                "hyp": name,
                "scheme": scheme,
                "status": res.status.value,
                "cv": res.mean_score,
                "cv_std": res.std_score,
                "folds": [round(f, 3) for f in res.fold_scores],
                "cv_seconds": round(time.time() - t, 1),
                "best_iter": h.params.get("_best_iteration"),
                "train_fraction": h.params.get("_train_fraction"),
            }
            if res.mean_score is not None:
                # финал как в агенте и без поправки #218 (как будто доля train в фолде 0.8)
                for variant, fraction in (("as_agent", h.params.get("_train_fraction")), ("no_scale", 0.8)):
                    hv = h.model_copy(update={"params": {**h.params, "_train_fraction": fraction}})
                    pred = fit_predict_final(spec, profile, hv, train, test)[spec.target]
                    row[f"honest_{variant}"] = round(rmse(y_hold, pred), 3)
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
