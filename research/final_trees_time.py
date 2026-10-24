"""Сколько деревьев брать в финале на схеме time: (#218): _best_iteration из CV или с поправкой на долю train.

    uv run python research/final_trees_time.py [--task tasks/public/regression/task.md] [--models lightgbm xgboost] [--params JSON]

Честная часть — последний период train (как bench_playbook.honest_split), train-часть — остальное. На train-части —
run_experiment на схеме choose_validation (как у агента), затем fit_predict_final-подобное обучение на всей
train-части с запасом деревьев и прогноз отложенной части на разных числах деревьев (префиксы одной модели =
модель, обученная ровно на столько деревьев). Сравниваются:
  cv      — _best_iteration × FINAL_TREES_MARGIN (как до #218);
  scaled  — то же × _trees_scale (как сейчас в fit_predict_final);
  last    — лучшая итерация последнего фолда (train ближе всего к финальному) × FINAL_TREES_MARGIN;
  optimum — лучшее число деревьев на честной части (подглядывание — только как ориентир).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ml.experiment as mx  # noqa: E402
from agent.task_parser import parse_task  # noqa: E402
from evaluation.metrics import compute_metric  # noqa: E402
from interfaces.schemas import Hypothesis, HypothesisKind, RunBudget  # noqa: E402
from ml.profiling import load_data, make_matrix, profile_data  # noqa: E402
from ml.validation import choose_validation  # noqa: E402
from research.bench_playbook import honest_split  # noqa: E402


def _staged(model, x, n: int, kind: str) -> np.ndarray:
    """Прогноз первых n деревьев одной обученной модели."""
    if kind == "lightgbm":
        return model.predict(x, num_iteration=n)
    if kind == "xgboost":
        return model.predict(x, iteration_range=(0, n))
    return model.predict(x, ntree_end=n)  # catboost


def run(task_md: Path, models: list[str], extra: dict) -> list[dict]:
    spec = parse_task(task_md)
    train, test = load_data(spec)
    profile = profile_data(spec, train, test)
    tr, va, how = honest_split(spec, profile, train, test)
    part, hold = train.iloc[tr].reset_index(drop=True), train.iloc[va].reset_index(drop=True)
    hold_x = hold.drop(columns=[spec.target])
    part_profile = profile_data(spec, part, hold_x)
    validation = choose_validation(spec, part_profile, part, hold_x)
    print(f"== {task_md.parent.name}: {spec.metric.name}, честно — {how} ({len(part)}/{len(hold)}), CV — {validation}")

    captured: list = []
    original = mx._mean_best_iteration

    def capture(best_iters):
        captured.append(list(best_iters))
        return original(best_iters)

    mx._mean_best_iteration = capture
    rows = []
    for model_name in models:
        params = {"model": model_name, "validation": validation, **copy.deepcopy(extra)}
        hyp = Hypothesis(id=model_name, kind=HypothesisKind.MODEL, title=model_name, params=params)
        captured.clear()
        res = mx.run_experiment(spec, part_profile, hyp, part, RunBudget())
        per_fold = captured[-1]
        best, fraction = hyp.params["_best_iteration"], hyp.params["_train_fraction"]
        scale = mx._trees_scale(hyp.params)
        variants = {
            "cv": mx._final_estimators(best),
            "scaled": mx._final_estimators(best, scale),
            "last": mx._final_estimators(per_fold[-1]),
        }
        # Обучение на всей train-части с запасом деревьев: префиксы — модели с меньшим числом деревьев
        log_target = mx._use_log_target(spec, hyp.params)
        x, x_hold = make_matrix(part_profile, part, hold_x)
        y = mx._encode_target(spec, part[spec.target])
        y_fit = np.log1p(np.clip(y, 0, None)) if log_target else y
        n_max = int(max(variants.values()) * 1.3)
        model, _ = mx._fit_model(spec, hyp, x, y_fit, n_estimators=n_max)

        grid = sorted({max(10, int(n_max * q / 40)) for q in range(1, 41)} | set(variants.values()))
        y_hold = hold[spec.target].to_numpy()
        curve = {}
        for n in grid:
            pred = _staged(model, x_hold, n, model_name)
            curve[n] = compute_metric(spec.metric.name, y_hold, np.expm1(pred) if log_target else pred)
        gib = spec.metric.greater_is_better
        opt_n = (max if gib else min)(curve, key=curve.get)
        row = {
            "model": model_name,
            "cv": res.mean_score,
            "per_fold": per_fold,
            "fraction": fraction,
            "scale": scale,
            **{f"n_{k}": v for k, v in variants.items()},
            **{f"s_{k}": curve[v] for k, v in variants.items()},
            "n_opt": opt_n,
            "s_opt": curve[opt_n],
        }
        rows.append(row)
        print(
            f"{model_name}: CV {res.mean_score:.3f}, деревья по фолдам {per_fold}, доля train {fraction}, ×{scale:.2f}\n"
            f"  cv     {variants['cv']:5d} деревьев → {curve[variants['cv']]:.3f}\n"
            f"  scaled {variants['scaled']:5d} деревьев → {curve[variants['scaled']]:.3f}\n"
            f"  last   {variants['last']:5d} деревьев → {curve[variants['last']]:.3f}\n"
            f"  optimum {opt_n:4d} деревьев → {curve[opt_n]:.3f} (подглядывание)\n"
            f"  кривая: " + ", ".join(f"{n}:{s:.2f}" for n, s in list(curve.items())[:: max(1, len(curve) // 12)])
        )
    mx._mean_best_iteration = original
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=Path, default=Path("tasks/public/regression/task.md"))
    parser.add_argument("--models", nargs="+", default=["lightgbm", "xgboost", "catboost"])
    parser.add_argument("--params", default="{}", help="общие params гипотезы, JSON: '{\"log_target\": true}'")
    args = parser.parse_args()
    run(args.task, args.models, json.loads(args.params))


if __name__ == "__main__":
    main()
