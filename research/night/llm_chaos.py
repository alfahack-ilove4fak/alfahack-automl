"""Ночной аудит: агент против плохой LLM (поддельный сервер research/night/fake_llm_thenlaooo.py).

    PYTHONPATH=. uv run python research/night/llm_chaos.py --task tasks/practice/cancer_binary_auc/task.md --budget 600 \
        --modes garbage,badjson,... --out /abs/path
Для каждого режима: свой порт и свой LLM_CACHE_DIR, агент с LLM_MOCK=0 и ключом «fake». Метрики прогона: время, код выхода,
валидность submission.csv, число запросов к серверу по назначению, ошибки в логе агента, итог цикла.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LLM_MOCK", "1")  # для parse_task в этом процессе
from agent.task_parser import parse_task  # noqa: E402
from evaluation.submission import validate_submission  # noqa: E402

DELAYS = {"slow": 25, "hang": 400}


def run_mode(mode: str, port: int, task: Path, budget: int, out: Path) -> dict:
    run_dir = out / mode
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.copytree(task.parent, run_dir / "task", ignore=shutil.ignore_patterns("submission.csv"))
    server_log = (run_dir / "server.log").open("w")
    server = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "research/night/fake_llm_thenlaooo.py"),
            "--mode",
            mode,
            "--port",
            str(port),
            "--delay",
            str(DELAYS.get(mode, 0)),
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.5)
    env = {
        **os.environ,
        "LLM_MOCK": "0",
        "OPENAI_API_KEY": "fake",
        "OPENAI_MODEL": "fake",
        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "LLM_CACHE_DIR": str(run_dir / "llm_cache"),
        "AGENT_TIME_BUDGET_SEC": str(budget),
        "AGENT_RUN_DIR": str(run_dir),
        "AGENT_LLM_CRITIC": "1",
        "LLM_RUB_LIMIT": "0",
    }
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY"):
        env.pop(var, None)
    started = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "solution.py"), str(run_dir / "task/task.md")],
        env=env,
        cwd=run_dir / "task",
        stdout=(run_dir / "stdout.txt").open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        code = proc.wait(timeout=budget * 1.4 + 90)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        code = "OVERRUN"
    seconds = round(time.time() - started)
    server.kill()
    spec = parse_task(run_dir / "task/task.md")
    rep = (
        validate_submission(spec.submission_path, spec, pd.read_csv(spec.test_path))
        if spec.submission_path.exists()
        else None
    )
    log = (
        (run_dir / "agent.log").read_text(encoding="utf-8", errors="replace")
        if (run_dir / "agent.log").exists()
        else ""
    )
    served = [ln.split()[1] for ln in (run_dir / "server.log").read_text().splitlines() if "mode=" in ln]
    return {
        "mode": mode,
        "seconds": seconds,
        "exit": code,
        "valid": None if rep is None else rep.ok,
        "server_calls": {p: served.count(p) for p in sorted(set(served))},
        "iterations": log.count("Итерация"),
        "llm_errors": log.count("LLM ошибка"),
        "tracebacks": log.count("Traceback"),
        "summary": [
            ln[-150:] for ln in log.splitlines() if "LLM итог" in ln or "Решения LLM" in ln or "Готово за" in ln
        ][:3],
        "feature": [ln[-140:] for ln in log.splitlines() if "признак" in ln.lower() or "песочниц" in ln.lower()][:4],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for i, mode in enumerate(args.modes.split(",")):
        row = run_mode(mode, args.port + i, args.task.resolve(), args.budget, args.out)
        (args.out / "results.jsonl").open("a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
