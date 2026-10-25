"""Сиды финала (#261, E3): честный скор финала на 1, 3 и 5 сидах — одна CV, один набор деревьев.

    uv run python research/final_seeds.py --all                       # public + practice
    uv run python research/final_seeds.py --task tasks/public/regression/task.md --seeds 1 3 5

Для каждой задачи: честная отложенная часть (bench_playbook.honest_split — время или группы), бейзлайн агента,
run_experiment на train-части (чтобы получить _best_iteration и схему CV, как у агента), затем fit_predict_final
на всей train-части при AGENT_FINAL_SEEDS = 1, 3, 5 и метрика задачи на отложенной части. Отбор по отложенной части не
идёт — только сравнение одного и того же финала, поэтому правило «не подбирать по своей же проверке» не нарушено.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ml.experiment as mx  # noqa: E402
from agent.hypotheses import baseline_hypothesis  # noqa: E402
from agent.task_parser import parse_task  # noqa: E402
from evaluation.local_score import score_submission  # noqa: E402
from interfaces.schemas import ExperimentStatus, RunBudget  # noqa: E402
from ml.profiling import load_data, profile_data  # noqa: E402
from ml.validation import choose_validation  # noqa: E402
from research.bench_playbook import honest_split, with_validation  # noqa: E402


def run(task_md: Path, seeds: list[int]) -> dict | None:
    spec = parse_task(task_md, use_analyst=False)
    train, test = load_data(spec)
    profile = profile_data(spec, train, test)
    tr, va, how = honest_split(spec, profile, train, test)
    part, hold = train.iloc[tr].reset_index(drop=True), train.iloc[va].reset_index(drop=True)
    hold_x = hold.drop(columns=[spec.target])
    part_profile = profile_data(spec, part, hold_x)
    validation = choose_validation(spec, part_profile, part, hold_x)
    hyp = with_validation(baseline_hypothesis(spec), validation)
    res = mx.run_experiment(spec, part_profile, hyp, part, RunBudget())
    if res.status != ExperimentStatus.OK:
        print(f"{task_md.parent.name}: CV не посчиталась ({res.error})")
        return None
    row = {"task": task_md.parent.name, "metric": spec.metric.name, "how": how, "cv": res.mean_score}
    for n in seeds:
        os.environ[mx.FINAL_SEEDS_ENV] = str(n)
        began = time.time()
        answer = mx.fit_predict_final(spec, part_profile, hyp, part, hold_x)
        seconds = time.time() - began
        scored = score_submission(spec, hold[spec.target], answer[spec.target], part[spec.target])
        row[n] = (float(scored["value"]), seconds)
    greater = mx.METRICS[spec.metric.name].greater_is_better
    base = row[seeds[0]][0]
    print(
        f"{row['task']:34s} {spec.metric.name:18s} "
        + "  ".join(f"{n} сид: {row[n][0]:.5g} ({row[n][1]:.0f} с)" for n in seeds)
        + f"  | {'лучше' if (row[seeds[-1]][0] > base) == greater else 'хуже'} на {abs(row[seeds[-1]][0] / base - 1):.2%}"
        if base
        else row["task"],
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task", type=Path, action="append")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()
    tasks = (
        sorted(ROOT.glob("tasks/public/*/task.md")) + sorted(ROOT.glob("tasks/practice/*/task.md"))
        if args.all
        else args.task
    )
    rows = [r for r in (run(t.resolve(), args.seeds) for t in tasks or []) if r]
    if rows:
        better = sum(
            (r[args.seeds[-1]][0] > r[args.seeds[0]][0]) == mx.METRICS[r["metric"]].greater_is_better
            and r[args.seeds[-1]][0] != r[args.seeds[0]][0]
            for r in rows
        )
        print(f"\nсидов {args.seeds[-1]} лучше, чем {args.seeds[0]}: {better} из {len(rows)}")
        ratio = np.mean([r[args.seeds[-1]][1] / max(r[args.seeds[0]][1], 1e-9) for r in rows])
        print(f"время финала ×{ratio:.1f}")


if __name__ == "__main__":
    main()
