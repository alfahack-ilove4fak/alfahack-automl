"""#276, загадка regression: какие составляющие ответа сдвигают прогноз на test.

Код агента — из worktree версии (первый аргумент), задачи — папки с task.md (остальные аргументы).
Для каждой задачи обучаем бейзлайн LightGBM в нескольких вариантах (схема CV × цель × поправка деревьев),
финальная модель — на всём train задачи, как в агенте. Если рядом с task.md лежит ../answers.csv (харнесс),
считаем RMSE на отложенной части; иначе — только статистику прогноза.

    uv run python research/night/reg_variants.py <worktree> <папка задачи> [<папка задачи> ...]
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("AGENT_TIME_BUDGET_SEC", "36000")
os.environ["LLM_MOCK"] = "1"
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))

from agent.hypotheses import baseline_hypothesis  # noqa: E402
from agent.task_parser import parse_task  # noqa: E402
from ml.experiment import _final_estimators, _trees_scale, fit_predict_final, run_experiment  # noqa: E402
from ml.profiling import load_data, profile_data  # noqa: E402
from ml.validation import choose_validation  # noqa: E402
from runtime.budget import make_budget  # noqa: E402

POISSON = {"model_params": {"objective": "poisson"}}


def variants(validation: str) -> dict[str, dict]:
    return {
        "kfold_L2": {},
        "kfold_poisson": POISSON,
        "time_L2": {"validation": validation},
        "time_L2_trees_x1.1": {"validation": validation, "_no_scale": True},
        "time_poisson": {"validation": validation, **POISSON},
    }


def main() -> None:
    for folder in sys.argv[2:]:
        task_md = Path(folder).resolve() / "task.md"
        spec = parse_task(task_md)
        train, test = load_data(spec)
        profile = profile_data(spec, train, test)
        validation = choose_validation(spec, profile, train, test)
        answers = task_md.parent.parent / "answers.csv"
        truth = pd.read_csv(answers) if answers.exists() else None
        print(f"\n### {folder}: train {len(train)}, test {len(test)}, схема {validation}", flush=True)
        rows = []
        for name, extra in variants(validation).items():
            base = baseline_hypothesis(spec)
            params = {**base.params, **{k: v for k, v in extra.items() if k != "_no_scale"}}
            hyp = base.model_copy(update={"id": name, "params": params})
            result = run_experiment(spec, profile, hyp, train, make_budget())
            if extra.get("_no_scale"):
                hyp.params.pop("_train_fraction", None)  # как в A и №1: деревьев = best_iter × 1.1
            trees = _final_estimators(hyp.params.get("_best_iteration"), _trees_scale(hyp.params))
            pred = fit_predict_final(spec, profile, hyp, train, test)
            out = task_md.parent / f"variant_{name}.csv"
            pred.to_csv(out, index=False)
            p = pred[spec.target]
            row = {
                "variant": name,
                "cv": round(float(result.mean_score), 3) if result.mean_score is not None else None,
                "best_iter": hyp.params.get("_best_iteration"),
                "trees": trees,
                "pred_mean": round(float(p.mean()), 2),
                "pred_top20": round(float(p[p >= p.quantile(0.8)].mean()), 1),
            }
            if truth is not None:
                key = spec.id_column
                m = truth.astype({key: str}).merge(pred.astype({key: str}), on=key, suffixes=("_t", "_p"))
                yt, yp = m[f"{spec.target}_t"], m[f"{spec.target}_p"]
                row.update(
                    truth_mean=round(float(yt.mean()), 2), rmse=round(float(np.sqrt(((yt - yp) ** 2).mean())), 2)
                )
            rows.append(row)
            print(row, flush=True)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
