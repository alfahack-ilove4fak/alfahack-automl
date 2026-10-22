"""Защита от нуля: агент завис / убит / упал на середине цикла — submission.csv всё равно валиден.

Запускаем настоящий solution.py отдельным процессом на маленькой задаче, подменяя только цикл AutoResearch
(искусственное зависание или падение). Без сети и LLM.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from agent.task_parser import parse_task
from evaluation.submission import validate_submission

ROOT = Path(__file__).resolve().parents[2]

DRIVER = """
import sys, time
import solution

mode = sys.argv[2]


def broken_loop(*args, **kwargs):
    print("ЦИКЛ НАЧАЛСЯ", file=sys.stderr, flush=True)
    if mode == "sleep":  # зависание внутри C-вызова, как у долгого fit
        time.sleep(10**6)
    elif mode == "busy":  # зависание в чистом питоне
        while True:
            pass
    raise RuntimeError("искусственная ошибка в цикле")


solution.run_autoresearch = broken_loop
sys.exit(solution.main(["solution.py", sys.argv[1]]))
"""


def _make_task(dst: Path) -> Path:
    data = load_breast_cancer(as_frame=True).frame.rename(columns=lambda c: c.replace(" ", "_"))
    data.insert(0, "row_id", range(len(data)))
    data["diagnosis"] = data.pop("target").map({0: "malignant", 1: "benign"})
    dst.mkdir(parents=True)
    train, test = data.iloc[:450], data.iloc[450:].drop(columns="diagnosis")
    train.to_csv(dst / "train.csv", index=False)
    test.to_csv(dst / "test.csv", index=False)
    test[["row_id"]].assign(diagnosis=0.5).to_csv(dst / "sample_submission.csv", index=False)
    (dst / "task.md").write_text(
        "# Diagnosis\n\nPredict `diagnosis`. Files: `train.csv`, `test.csv`.\n\nМетрика — ROC-AUC, в ответе вероятность.\n",
        encoding="utf-8",
    )
    return dst / "task.md"


def _start(tmp_path: Path, mode: str, budget_sec: int, safety_sec: int) -> tuple[subprocess.Popen, Path]:
    task_md = _make_task(tmp_path / "task")
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",  # драйвер не проходит через main() solution.py, где stderr переводится в UTF-8
        "AGENT_TIME_BUDGET_SEC": str(budget_sec),
        "AGENT_SAFETY_SEC": str(safety_sec),
        "LLM_MOCK": "1",
        "AGENT_RUN_DIR": str(tmp_path / "run"),
    }
    proc = subprocess.Popen(
        [sys.executable, str(driver), str(task_md), mode],
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, task_md


def _assert_valid_submission(task_md: Path) -> None:
    spec = parse_task(task_md)
    report = validate_submission(spec.submission_path, spec, pd.read_csv(spec.test_path))
    assert report.ok, report.problems
    assert not spec.submission_path.with_suffix(".tmp.csv").exists()


@pytest.mark.parametrize("mode", ["sleep", "busy"])
def test_hang_in_loop_is_cut_by_watchdog(tmp_path, mode):
    """Цикл завис навсегда — сторож по жёсткому дедлайну проверяет ответ и завершает процесс до конца бюджета."""
    started = time.time()
    proc, task_md = _start(tmp_path, mode, budget_sec=15, safety_sec=5)
    try:
        out, _ = proc.communicate(timeout=60)
    finally:
        proc.kill()
    elapsed = time.time() - started
    assert proc.returncode == 0, out[-3000:]
    assert "Сторож сработал: жёсткий дедлайн" in out, out[-3000:]
    assert elapsed < 15 + 15, f"процесс жил {elapsed:.0f} с при бюджете 15 с"  # запас на импорты в начале
    _assert_valid_submission(task_md)


def test_sigterm_in_the_middle_of_loop(tmp_path):
    """Платформа (или docker stop) прислала SIGTERM, пока цикл завис, — ответ с бейзлайном остаётся валидным."""
    proc, task_md = _start(tmp_path, "sleep", budget_sec=600, safety_sec=90)
    try:
        deadline = time.time() + 90
        seen = ""
        while time.time() < deadline and "ЦИКЛ НАЧАЛСЯ" not in seen:
            assert proc.poll() is None, f"процесс завершился сам: {seen[-2000:]}"
            line = proc.stdout.readline()
            seen += line
        assert "ЦИКЛ НАЧАЛСЯ" in seen, "до цикла дошли не вовремя"
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        proc.kill()
    seen += out
    assert proc.returncode == 128 + signal.SIGTERM, seen[-3000:]
    assert "Сторож сработал: сигнал SIGTERM" in seen, seen[-3000:]
    assert "Сохранён ответ «baseline»" in seen, seen[-3000:]
    _assert_valid_submission(task_md)


def test_exception_in_loop_keeps_answer_and_exit_code_zero(tmp_path):
    proc, task_md = _start(tmp_path, "raise", budget_sec=60, safety_sec=10)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, out[-3000:]
    assert "Агент упал" in out
    assert "Сохранён ответ «baseline»" in out, out[-3000:]
    _assert_valid_submission(task_md)
