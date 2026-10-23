"""Агент против плохой LLM: поддельный сервер (research/night/fake_llm.py) в разных режимах. Ночной аудит, фаза 4.

    uv run python research/night/llm_faults.py --modes garbage badcode --budget 180
    uv run python research/night/llm_faults.py --modes hang --budget 10800 --kill-after 720   # 3 ч, смотрим 12 мин

Для каждого режима: копия задачи (по умолчанию tasks/practice/cancer_binary_auc), solution.py с OPENAI_BASE_URL на
localhost, без LLM_MOCK. Пишем: время, код выхода, когда записан шаг 0 (тривиальный ответ) и бейзлайн,
сколько итераций цикла, сколько запросов к LLM и ошибок, итог «Решения LLM», ROC-AUC ответа по _answers.
Результат → runs/night/llm_faults.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "night" / "llm_faults"
TASK = ROOT / "tasks" / "practice" / "cancer_binary_auc"
ANSWERS = ROOT / "tasks" / "practice" / "_answers" / "cancer_binary_auc.csv"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def first_time(text: str, pattern: str) -> float | None:
    m = re.search(r"\[\s*([0-9.]+)s\][^\n]*" + pattern, text)
    return float(m.group(1)) if m else None


def run_mode(mode: str, budget: int, kill_after: float | None) -> dict:
    run_dir = OUT / f"{mode}-{budget}"
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.copytree(TASK, run_dir / "task", ignore=shutil.ignore_patterns("submission.csv"))
    port = free_port()
    server = None
    if mode != "refused":  # refused — порт закрыт, сервера нет («LLM недоступна»)
        server = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("fake_llm.py")), mode, str(port)],
            stderr=open(run_dir / "fake_llm.log", "w"),
        )
        time.sleep(1.5)
    env = {
        **os.environ,
        "OPENAI_API_KEY": "fake-key",
        "OPENAI_MODEL": "fake-model",
        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "AGENT_TIME_BUDGET_SEC": str(budget),
        "AGENT_RUN_DIR": str(run_dir),
    }
    env.pop("LLM_MOCK", None)
    started = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "solution.py"), str(run_dir / "task" / "task.md")],
        cwd=run_dir / "task",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=open(run_dir / "stderr.txt", "w"),
    )
    killed = False
    while proc.poll() is None:
        if kill_after and time.time() - started > kill_after:
            proc.send_signal(signal.SIGKILL)
            proc.wait()
            killed = True
            break
        time.sleep(1)
    seconds = time.time() - started
    if server:
        server.kill()
    log = (
        (run_dir / "agent.log").read_text(encoding="utf-8", errors="replace")
        if (run_dir / "agent.log").exists()
        else ""
    )
    row = {
        "mode": mode,
        "budget": budget,
        "seconds": round(seconds, 1),
        "exit_code": proc.returncode,
        "killed_by_us": killed,
        "step0_at": first_time(log, "Шаг 0"),
        "baseline_at": first_time(log, "Шаг 1"),
        "iterations": len(re.findall(r"Итерация \d+:", log)),
        "llm_errors_logged": len(re.findall(r"LLM ошибка", log)),
        "llm_summary": (re.findall(r"LLM итог: (.*)", log) or [None])[-1],
        "llm_decisions": (re.findall(r"Решения LLM: (.*)", log) or [None])[-1],
        "tracebacks": log.count("Traceback"),
        "server_requests": sum(1 for _ in open(run_dir / "fake_llm.log")) - 1 if server else 0,
    }
    sub_path = run_dir / "task" / "submission.csv"
    if sub_path.exists():
        try:
            sub = pd.read_csv(sub_path)
            ans = pd.read_csv(ANSWERS)
            merged = ans.merge(sub, on=ans.columns[0], suffixes=("_true", "_pred"))
            t = ans.columns[1]
            row["rows_ok"] = len(merged) == len(ans) == len(sub)
            row["auc"] = round(float(roc_auc_score(merged[f"{t}_true"], merged[f"{t}_pred"])), 5)
        except Exception as exc:  # noqa: BLE001
            row["sub_error"] = f"{type(exc).__name__}: {exc}"[:200]
    else:
        row["sub_error"] = "нет submission.csv"
    with (ROOT / "runs" / "night" / "llm_faults.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=180)
    parser.add_argument("--kill-after", type=float)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        run_mode(mode, args.budget, args.kill_after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
