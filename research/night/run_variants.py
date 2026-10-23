"""Прогон агента на вариантах формата (gen_variants.py), бюджет 60 с: что вышло в submission.csv.

    python research/night/run_variants.py runs/night/variants
"""

from __future__ import annotations

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


def main(root: Path) -> None:
    exp = json.loads((root / "_expected.json").read_text())
    for name, e in exp.items():
        run = ROOT / "runs" / "night" / "work_v" / name
        shutil.rmtree(run, ignore_errors=True)
        shutil.copytree(root / name, run / "task")
        env = {**os.environ, "AGENT_TIME_BUDGET_SEC": "60", "AGENT_RUN_DIR": str(run), "LLM_MOCK": "1"}
        t0 = time.time()
        p = subprocess.run([sys.executable, str(ROOT / "solution.py"), str(run / "task" / "task.md")], env=env,
                           cwd=run, capture_output=True, text=True, timeout=300)
        sec = round(time.time() - t0)
        sub = run / "task" / "submission.csv"
        ans = pd.read_csv(root / "_answers" / f"{name}.csv")
        res = {"task": name, "note": e["note"], "exit": p.returncode, "sec": sec, "sub": sub.exists()}
        if sub.exists():
            s = pd.read_csv(sub)
            res["cols"] = list(s.columns)[:4]
            res["rows"] = len(s)
            ok = list(s.columns) == ["id", e["target"]] and len(s) == len(ans)
            res["format_ok"] = ok
            if ok:
                mae = float(np.mean(np.abs(pd.to_numeric(s[e["target"]], errors="coerce") - ans[e["target"]].to_numpy())))
                res["mae"] = round(mae, 4)
                res["const0_mae"] = round(float(np.mean(np.abs(ans[e["target"]]))), 4)
        log = run / "agent.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        first = [ln for ln in text.splitlines() if "ERROR" in ln][:1]
        res["first_error"] = first[0][:200] if first else ""
        print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
