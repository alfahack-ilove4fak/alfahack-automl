"""Запустить агента на задаче «как на платформе», не засоряя tasks/: задача копируется в runs/.

    uv run python scripts/run_local.py tasks/public/regression/task.md --minutes 10
    uv run python scripts/run_local.py --all --minutes 5        # все задачи из tasks/public и tasks/practice

Печатает время, итог проверки submission.csv и путь к логу.
Харнесс с подсчётом баллов на отложенной части train — задача QA (evaluation/harness.py).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_one(task_md: Path, minutes: float, mock_llm: bool) -> bool:
    import pandas as pd

    from agent.task_parser import parse_task
    from evaluation.submission import validate_submission

    name = task_md.parent.relative_to(ROOT).as_posix().replace("/", "__")
    run_dir = ROOT / "runs" / f"{time.strftime('%m%d-%H%M%S')}-{name}"
    shutil.copytree(task_md.parent, run_dir / "task", ignore=shutil.ignore_patterns("submission.csv"))
    env = {**os.environ, "AGENT_TIME_BUDGET_SEC": str(int(minutes * 60)), "AGENT_RUN_DIR": str(run_dir)}
    if mock_llm:
        env["LLM_MOCK"] = "1"
    started = time.time()
    proc = subprocess.run([sys.executable, str(ROOT / "solution.py"), str(run_dir / "task" / task_md.name)],
                          env=env, cwd=run_dir)
    spec = parse_task(run_dir / "task" / task_md.name)
    report = validate_submission(spec.submission_path, spec, pd.read_csv(spec.test_path))
    status = "✅" if proc.returncode == 0 and report.ok else "❌"
    print(f"{status} {name}: {time.time() - started:.0f} с, код выхода {proc.returncode}, "
          f"сабмит {'валиден' if report.ok else report.problems}\n   лог: {run_dir / 'agent.log'}")
    return status == "✅"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--mock-llm", action="store_true", help="без LLM (не тратит бюджет)")
    args = parser.parse_args()
    if args.all:
        tasks = sorted((ROOT / "tasks").glob("*/*/task.md"))
    elif args.task:
        tasks = [args.task.resolve()]
    else:
        parser.error("укажи путь к task.md или --all")
    results = [run_one(t, args.minutes, args.mock_llm) for t in tasks]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
