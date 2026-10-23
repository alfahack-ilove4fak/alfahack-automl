"""Ночной аудит: прогон агента на синтетических задачах как на платформе + независимая оценка.

    uv run python research/night/run_tasks.py --minutes 5 [--only t01] [--out runs/night/p3.jsonl]

Задача копируется в runs/night/p3/<имя>/ (только папка задачи — ответы агент не видит), solution.py запускается
с LLM_MOCK=1 и AGENT_TIME_BUDGET_SEC. Пик RAM — опрос psutil по дереву процессов раз в 0.5 с.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "night"))
from eval_tasks import evaluate  # noqa: E402


def run(task: Path, minutes: float, out_root: Path) -> dict:
    run_dir = out_root / task.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(task, run_dir / "task")
    env = {
        **os.environ,
        "AGENT_TIME_BUDGET_SEC": str(int(minutes * 60)),
        "AGENT_RUN_DIR": str(run_dir),
        "LLM_MOCK": "1",
    }
    started = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "solution.py"), str(run_dir / "task" / "task.md")],
        env=env,
        cwd=run_dir / "task",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    peak = 0
    p = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
            peak = max(peak, rss)
        except psutil.Error:
            pass
        time.sleep(0.5)
    err = proc.stderr.read().decode("utf-8", "replace")
    res = {"seconds": round(time.time() - started), "exit": proc.returncode, "peak_mb": round(peak / 2**20)}
    try:
        res.update(evaluate(run_dir / "task", task.parent / "_answers" / task.name))
    except Exception as exc:  # noqa: BLE001
        res.update(task=task.name, eval_error=f"{type(exc).__name__}: {exc}"[:200])
    log = (
        (run_dir / "agent.log").read_text(encoding="utf-8", errors="replace")
        if (run_dir / "agent.log").exists()
        else ""
    )
    res["log_errors"] = log.count(" ERROR ") + log.count("Traceback")
    res["stderr_tail"] = err[-300:]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="runs/night/tasks")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="runs/night/p3.jsonl")
    args = ap.parse_args()
    tasks_root = ROOT / args.tasks
    out_root = ROOT / "runs" / "night" / "p3"
    for task in sorted(p for p in tasks_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if args.only and not any(o in task.name for o in args.only.split(",")):
            continue
        res = run(task, args.minutes, out_root)
        print(json.dumps(res, ensure_ascii=False, default=float), flush=True)
        with open(ROOT / args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False, default=float) + "\n")


if __name__ == "__main__":
    main()
