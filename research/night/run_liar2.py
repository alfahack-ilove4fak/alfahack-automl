"""Мок-LLM лжёт аналитику задачи (target=f0, id=cat_a, metric=mae) на задаче без шаблона и без метрики в тексте."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
run = ROOT / "runs/night/work_liar2"
shutil.rmtree(run, ignore_errors=True)
shutil.copytree(ROOT / "runs/night/tasks_liar/t12b", run / "task")
srv = subprocess.Popen([sys.executable, str(ROOT / "research/night/mock_llm.py"), "18200", "liar2"])
time.sleep(1.5)
env = {
    **os.environ,
    "AGENT_TIME_BUDGET_SEC": "240",
    "AGENT_RUN_DIR": str(run),
    "OPENAI_API_KEY": "sk",
    "OPENAI_MODEL": "mock",
    "OPENAI_BASE_URL": "http://127.0.0.1:18200/v1",
    "LLM_CACHE_DIR": str(run / "cache"),
}
env.pop("LLM_MOCK", None)
p = subprocess.run(
    [sys.executable, str(ROOT / "solution.py"), str(run / "task/task.md")],
    env=env,
    cwd=run,
    capture_output=True,
    text=True,
)
srv.kill()
ans = pd.read_csv(ROOT / "runs/night/tasks/_answers/t12_ordinal_qwk_nosample.csv")
s = pd.read_csv(run / "task/submission.csv")
print(
    "exit",
    p.returncode,
    "cols",
    list(s.columns),
    "rows",
    len(s),
    "qwk",
    round(cohen_kappa_score(ans.iloc[:, -1], s.iloc[:, -1], weights="quadratic"), 4),
)
for ln in (run / "agent.log").read_text().splitlines():
    if "analyst" in ln or "Задача:" in ln or "task_analyst" in ln or "аналитик" in ln:
        print(ln[:220])
