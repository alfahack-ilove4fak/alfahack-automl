"""Ночной аудит: kill -9 агента в разные моменты — остаётся ли валидный submission.csv.

    PYTHONPATH=. uv run python research/night/kill9_check.py tasks/practice/cancer_binary_auc/task.md 2 5 10 20 40
Для каждой задержки (с) запускаем solution.py в новой копии задачи, через N секунд убиваем процесс -9 (вся группа) и проверяем файл.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("LLM_MOCK", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.task_parser import parse_task  # noqa: E402
from evaluation.submission import validate_submission  # noqa: E402

task = Path(sys.argv[1]).resolve()
for delay in map(float, sys.argv[2:]):
    work = Path(tempfile.mkdtemp(prefix="kill9-")) / "task"
    shutil.copytree(task.parent, work, ignore=shutil.ignore_patterns("submission.csv"))
    env = {**os.environ, "AGENT_TIME_BUDGET_SEC": "600", "AGENT_RUN_DIR": str(work.parent / "run")}
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "solution.py"), str(work / "task.md")],
        env=env,
        cwd=work,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(delay)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    sub = work / "submission.csv"
    tmp = list(work.glob("*.tmp*"))
    if not sub.exists():
        print(f"kill через {delay:5.1f} с: submission.csv НЕТ; хвосты записи: {[t.name for t in tmp]}")
        continue
    spec = parse_task(work / "task.md")
    rep = validate_submission(sub, spec, pd.read_csv(spec.test_path))
    print(
        f"kill через {delay:5.1f} с: файл есть, валиден={rep.ok}, строк={rep.n_rows}, проблемы={rep.problems[:2]}, хвосты записи: {[t.name for t in tmp]}"
    )
