"""#206: почему на маленьких данных (diabetes, ~300 строк) агент хуже LightGBM по умолчанию.

    uv run python research/night/r206_small.py [--task tasks/practice/diabetes_regression_mae/task.md] [--splits 20]

Одна отложенная часть харнесса — ~77 строк, разница 48.98 против 47.60 MAE может быть шумом. Поэтому: --splits
случайных разбиений 75/25 всего train; на каждом — путь агента (run_experiment → fit_predict_final, как в
solution.py) для нескольких вариантов params и «LightGBM по умолчанию» (как сильный бейзлайн харнесса).
Итог: средняя метрика на отложенной части и попарная разница с LightGBM по умолчанию (сколько разбиений лучше).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.task_parser import parse_task  # noqa: E402
from evaluation.metrics import METRICS, compute_metric  # noqa: E402
from interfaces.schemas import Hypothesis, HypothesisKind  # noqa: E402
from ml.experiment import fit_predict_final, run_experiment  # noqa: E402
from ml.profiling import load_data, make_matrix, profile_data  # noqa: E402
from runtime.budget import make_budget  # noqa: E402

VARIANTS = {
    "agent_baseline": {"model": "lightgbm"},
    "leaves15": {"model": "lightgbm", "model_params": {"num_leaves": 15}},
    "leaves31": {"model": "lightgbm", "model_params": {"num_leaves": 31}},
    "no_subsample": {"model": "lightgbm", "model_params": {"subsample": 1.0, "colsample_bytree": 1.0}},
    "lr01": {"model": "lightgbm", "model_params": {"learning_rate": 0.1}},
    "defaults_like": {
        "model": "lightgbm",
        "model_params": {"learning_rate": 0.1, "num_leaves": 31, "subsample": 1.0, "colsample_bytree": 1.0},
    },
    "ens_lgb_xgb": {"ensemble": ["lightgbm", "xgboost"]},
}


def lgbm_default(spec, profile, train, test) -> np.ndarray:
    x, x_test = make_matrix(profile, train, test)
    return lgb.LGBMRegressor(verbose=-1, random_state=0, n_jobs=4).fit(x, train[spec.target]).predict(x_test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tasks/practice/diabetes_regression_mae/task.md")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="runs/206/small.jsonl")
    args = ap.parse_args()
    spec = parse_task(ROOT / args.task)
    full, _ = load_data(spec)
    gib = METRICS[spec.metric.name].greater_is_better
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    variants = {k: v for k, v in VARIANTS.items() if not args.only or k in args.only.split(",")}
    scores: dict[str, list[float]] = {k: [] for k in ["lgbm_default", *variants]}
    for seed in range(args.splits):
        hold = full.sample(frac=0.25, random_state=seed)
        train = full.drop(hold.index).reset_index(drop=True)
        hold = hold.reset_index(drop=True)
        y = hold[spec.target].to_numpy()
        test = hold.drop(columns=[spec.target])
        profile = profile_data(spec, train, test)
        scores["lgbm_default"].append(compute_metric(spec.metric.name, y, lgbm_default(spec, profile, train, test)))
        for name, params in variants.items():
            h = Hypothesis(id=name, kind=HypothesisKind.HYPERPARAMS, title=name, rationale="#206", params=dict(params))
            res = run_experiment(spec, profile, h, train, make_budget(time.time()))
            pred = fit_predict_final(spec, profile, h, train, test)[spec.target]
            scores[name].append(compute_metric(spec.metric.name, y, pred))
            row = {
                "seed": seed,
                "hyp": name,
                "cv": res.mean_score,
                "best_iter": h.params.get("_best_iteration"),
                "honest": scores[name][-1],
                "lgbm_default": scores["lgbm_default"][-1],
            }
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        print(f"разбиение {seed}: готово", flush=True)
    base = np.array(scores["lgbm_default"])
    print(f"\n{spec.metric.name} ({'больше' if gib else 'меньше'} — лучше), {args.splits} разбиений")
    print(f"{'вариант':16}{'среднее':>10}{'к LGBM умолч.':>15}{'лучше в':>10}")
    for name, vals in scores.items():
        v = np.array(vals)
        diff = (v - base) if gib else (base - v)  # > 0 — лучше LightGBM по умолчанию
        print(f"{name:16}{v.mean():10.3f}{diff.mean():+15.3f}{int((diff > 0).sum()):>7}/{len(v)}")


if __name__ == "__main__":
    main()
