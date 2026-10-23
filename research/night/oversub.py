"""Ночной аудит: насколько LightGBM агента замедляется, когда CPU занят другим процессом.

    uv run python research/night/oversub.py [--hogs 4]

Замер: CV бейзлайна агента (run_experiment, 5 фолдов) на tasks/practice/cancer_binary_auc и
claims_regression_mape_rowno — сначала без нагрузки, затем рядом с --hogs процессами «пустой цикл» на всех ядрах
и рядом с вторым таким же LightGBM-процессом (как две задачи параллельно на одной машине).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def hog(stop):
    while not stop.is_set():
        pass


def baseline_seconds(task: str) -> float:
    from agent.hypotheses import baseline_hypothesis
    from agent.task_parser import parse_task
    from ml.experiment import run_experiment
    from ml.profiling import load_data, profile_data
    from runtime.budget import make_budget

    spec = parse_task(ROOT / task)
    train, test = load_data(spec)
    profile = profile_data(spec, train, test)
    t = time.time()
    res = run_experiment(spec, profile, baseline_hypothesis(), train, make_budget(time.time()))
    return round(time.time() - t, 1), res.status.value


def twin(task, q):
    q.put(baseline_seconds(task))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hogs", type=int, default=4)
    args = ap.parse_args()
    for task in ("tasks/practice/cancer_binary_auc/task.md", "tasks/practice/claims_regression_mape_rowno/task.md"):
        print(task, "без нагрузки:", baseline_seconds(task), flush=True)
        stop = mp.Event()
        procs = [mp.Process(target=hog, args=(stop,)) for _ in range(args.hogs)]
        for p in procs:
            p.start()
        print(task, f"рядом {args.hogs} пустых цикла:", baseline_seconds(task), flush=True)
        stop.set()
        for p in procs:
            p.join()
        q = mp.Queue()
        other = mp.Process(target=twin, args=(task, q))
        other.start()
        mine = baseline_seconds(task)
        other.join()
        print(task, "два агента одновременно:", mine, q.get(), flush=True)


if __name__ == "__main__":
    main()
