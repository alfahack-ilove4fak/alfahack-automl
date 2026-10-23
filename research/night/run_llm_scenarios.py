"""Прогон агента на задаче с «плохой» LLM (мок-сервер mock_llm.py) и без неё. Печатает JSON-строки.

python research/night/run_llm_scenarios.py --minutes 5 [modes...]
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks" / "practice" / "claims_regression_mape_rowno"
ANS = ROOT / "tasks" / "practice" / "_answers" / "claims_regression_mape_rowno.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument(
        "modes",
        nargs="*",
        default=["none", "down", "garbage", "slow", "badcode", "memory", "keyerr", "slow_code", "liar", "goodish"],
    )
    a = ap.parse_args()
    ans = pd.read_csv(ANS)
    port = 18080
    for mode in a.modes:
        run = ROOT / "runs" / "night" / "work_llm" / mode
        shutil.rmtree(run, ignore_errors=True)
        shutil.copytree(TASK, run / "task")
        env = {**os.environ, "AGENT_TIME_BUDGET_SEC": str(int(a.minutes * 60)), "AGENT_RUN_DIR": str(run)}
        for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "LLM_MOCK", "LLM_CACHE_DIR"):
            env.pop(k, None)
        srv = None
        if mode == "none":
            env["LLM_MOCK"] = "1"
        else:
            port += 1
            env.update(
                OPENAI_API_KEY="sk-test",
                OPENAI_MODEL="mock",
                OPENAI_BASE_URL=f"http://127.0.0.1:{port}/v1",
                LLM_CACHE_DIR=str(run / "cache"),
                LLM_RUB_LIMIT="0",
            )
            if mode != "down":
                srv = subprocess.Popen([sys.executable, str(ROOT / "research/night/mock_llm.py"), str(port), mode])
                time.sleep(1.5)
        t0 = time.time()
        p = subprocess.Popen(
            [sys.executable, str(ROOT / "solution.py"), str(run / "task" / "task.md")],
            env=env,
            cwd=run,
            stdout=(run / "out.txt").open("w"),
            stderr=subprocess.STDOUT,
        )
        try:
            p.wait(timeout=a.minutes * 60 + 90)
            hung = False
        except subprocess.TimeoutExpired:
            p.kill()
            hung = True
        sec = round(time.time() - t0)
        if srv:
            srv.kill()
        sub = run / "task" / "submission.csv"
        res = {
            "mode": mode,
            "sec": sec,
            "budget": int(a.minutes * 60),
            "exit": p.returncode,
            "hung": hung,
            "sub": sub.exists(),
        }
        if sub.exists():
            s = pd.read_csv(sub)
            ok = len(s) == len(ans) and s.iloc[:, -1].notna().all()
            res["format_ok"] = bool(ok)
            if ok:
                res["mape"] = round(
                    float(
                        np.mean(
                            np.abs(s.iloc[:, -1].to_numpy() - ans.iloc[:, -1].to_numpy())
                            / np.abs(ans.iloc[:, -1].to_numpy())
                        )
                    ),
                    4,
                )
        log = (run / "agent.log").read_text(errors="replace") if (run / "agent.log").exists() else ""
        res["llm_err"] = log.count("LLM ошибка")
        res["sandbox_msgs"] = [ln[:170] for ln in log.splitlines() if "песочниц" in ln.lower()][:3]
        res["best"] = [ln[ln.find("Лучшая") :][:80] for ln in log.splitlines() if "Лучшая гипотеза" in ln][:1]
        res["llm_summary"] = [ln[ln.find("LLM итог") :][:140] for ln in log.splitlines() if "LLM итог" in ln][:1]
        print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
