"""Прогон агента на big3m с заданным бюджетом; печатает время, пик RAM, RMSE на скрытых ответах."""
import os, shutil, subprocess, sys, time, threading, json
from pathlib import Path
import numpy as np, pandas as pd, psutil
ROOT = Path(__file__).resolve().parents[2]
minutes = float(sys.argv[1])
run = ROOT / "runs/night/work_big"
shutil.rmtree(run, ignore_errors=True)
run.mkdir(parents=True)
os.symlink(ROOT / "runs/night/tasks_big/big3m", run / "task")
# симлинк: копировать 2 ГБ незачем; submission.csv ляжет в исходную папку — это ок
env = {**os.environ, "AGENT_TIME_BUDGET_SEC": str(int(minutes * 60)), "AGENT_RUN_DIR": str(run), "LLM_MOCK": "1"}
t0 = time.time()
p = subprocess.Popen([sys.executable, str(ROOT / "solution.py"), str(run / "task/task.md")], env=env, cwd=run, stdout=open(run / "out.txt", "w"), stderr=subprocess.STDOUT)
peak = 0
while p.poll() is None:
    try:
        pr = psutil.Process(p.pid)
        peak = max(peak, pr.memory_info().rss + sum(c.memory_info().rss for c in pr.children(recursive=True)))
    except psutil.Error:
        pass
    time.sleep(1)
sub = pd.read_csv(ROOT / "runs/night/tasks_big/big3m/submission.csv")
ans = pd.read_csv(ROOT / "runs/night/tasks_big/big3m_answers.csv")
rmse = float(np.sqrt(np.mean((sub.iloc[:, -1].to_numpy() - ans["y"].to_numpy()) ** 2)))
print(json.dumps({"budget_s": int(minutes * 60), "sec": round(time.time() - t0), "exit": p.returncode, "peak_ram_mb": round(peak / 1e6), "rmse": round(rmse, 4), "std_y": round(float(ans["y"].std()), 4), "rows": len(sub), "distinct": int(sub.iloc[:, -1].nunique())}))
