"""Крайние условия платформы: крошечный бюджет времени и лимит памяти (фаза 4 ночного аудита).

    python research/night/run_small.py
Печатает JSON-строки: время, код выхода, валиден ли ответ, чем он стал (модель или тривиальный).
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCEN = [
    ("reg_30s", "tasks/public/regression", 30, None),
    ("reg_120s", "tasks/public/regression", 120, None),
    ("multi_30s", "tasks/public/multiclass", 30, None),
    ("big1m_ram3g_120s", "runs/night/tasks/t04_big_1m_mape", 120, 3 * 1024**3),
    ("big1m_ram1500m_300s", "runs/night/tasks/t04_big_1m_mape", 300, 1500 * 1024**2),
    ("wide_ram2g_120s", "runs/night/tasks/t05_wide_1500_prauc", 120, 2 * 1024**3),
]


def main() -> None:
    from agent.task_parser import parse_task
    from evaluation.submission import validate_submission

    for name, tdir, budget, limit in SCEN:
        run = ROOT / "runs" / "night" / "work_small" / name
        shutil.rmtree(run, ignore_errors=True)
        shutil.copytree(ROOT / tdir, run / "task", ignore=shutil.ignore_patterns("submission.csv", "_*"))
        env = {**os.environ, "AGENT_TIME_BUDGET_SEC": str(budget), "AGENT_RUN_DIR": str(run), "LLM_MOCK": "1"}

        def preexec(limit=limit):
            if limit:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        t0 = time.time()
        p = subprocess.Popen([sys.executable, str(ROOT / "solution.py"), str(run / "task" / "task.md")], env=env, cwd=run,
                             stdout=(run / "out.txt").open("w"), stderr=subprocess.STDOUT, preexec_fn=preexec)
        try:
            p.wait(timeout=budget + 120)
            hung = False
        except subprocess.TimeoutExpired:
            p.kill()
            hung = True
        sec = round(time.time() - t0)
        res = {"name": name, "budget": budget, "sec": sec, "exit": p.returncode, "hung": hung}
        try:
            spec = parse_task(run / "task" / "task.md")
            te = pd.read_csv(spec.test_path)
            rep = validate_submission(spec.submission_path, spec, te)
            res["valid"] = rep.ok
            res["problems"] = rep.problems[:2]
            if spec.submission_path.exists():
                col = pd.read_csv(spec.submission_path).iloc[:, -1]
                res["distinct_values"] = int(col.nunique())
        except Exception as exc:  # noqa: BLE001
            res["check_error"] = f"{type(exc).__name__}: {exc}"[:150]
        log = (run / "agent.log").read_text(errors="replace") if (run / "agent.log").exists() else ""
        res["saved"] = [ln.split("«")[1].split("»")[0] for ln in log.splitlines() if "Сохранён ответ" in ln][-1:]
        res["errors"] = [ln[:160] for ln in log.splitlines() if " ERROR " in ln][:2]
        res["watchdog"] = "Сторож сработал" in log
        print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
