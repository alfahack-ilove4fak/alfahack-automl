"""Дымовой тест: solution.py проходит путь от task.md до валидного submission.csv.

Запускаем как платформа — отдельным процессом — на маленьких копиях задач, с бюджетом 40 секунд.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from agent.task_parser import parse_task
from evaluation.submission import validate_submission

ROOT = Path(__file__).resolve().parents[2]


def _mini_copy(src: Path, dst: Path, n_train: int = 3000, n_test: int = 500) -> Path:
    dst.mkdir(parents=True)
    shutil.copy(src / "task.md", dst / "task.md")
    shutil.copy(src / "data_dictionary.csv", dst / "data_dictionary.csv")
    pd.read_csv(src / "train.csv", nrows=n_train).to_csv(dst / "train.csv", index=False)
    test = pd.read_csv(src / "test.csv", nrows=n_test)
    test.to_csv(dst / "test.csv", index=False)
    pd.read_csv(src / "sample_submission.csv", nrows=n_test).to_csv(dst / "sample_submission.csv", index=False)
    return dst / "task.md"


def _hidden_like_task(dst: Path) -> Path:
    """«Скрытая» задача другого типа: бинарная классификация с ROC-AUC, файлы во вложенной папке."""
    data = load_breast_cancer(as_frame=True).frame.rename(columns=lambda c: c.replace(" ", "_"))
    data.insert(0, "row_id", range(len(data)))
    data["diagnosis"] = data.pop("target").map({0: "malignant", 1: "benign"})
    files = dst / "data" / "files"
    files.mkdir(parents=True)
    train, test = data.iloc[:450], data.iloc[450:].drop(columns="diagnosis")
    train.to_csv(files / "train.csv", index=False)
    test.to_csv(files / "test.csv", index=False)
    test[["row_id"]].assign(diagnosis=0.5).to_csv(files / "sample_submission.csv", index=False)
    (dst / "task.md").write_text(
        "# Diagnosis\n\nPredict `diagnosis`. Files: `data/files/train.csv`, `data/files/test.csv`.\n\n"
        "Метрика — ROC-AUC, в ответе вероятность.\n",
        encoding="utf-8",
    )
    return dst / "task.md"


def _run_solution(task_md: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "AGENT_TIME_BUDGET_SEC": "40",
        "AGENT_SAFETY_SEC": "5",
        "LLM_MOCK": "1",
        "AGENT_RUN_DIR": str(tmp_path / "run"),
    }
    return subprocess.run(
        [sys.executable, str(ROOT / "solution.py"), str(task_md)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        cwd=tmp_path,
    )


@pytest.mark.parametrize("builder", ["regression", "multiclass", "hidden_binary"])
def test_solution_end_to_end(builder, tmp_path):
    if builder == "hidden_binary":
        task_md = _hidden_like_task(tmp_path / "task")
    else:
        task_md = _mini_copy(ROOT / "tasks" / "public" / builder, tmp_path / "task")
    proc = _run_solution(task_md, tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    spec = parse_task(task_md)
    report = validate_submission(spec.submission_path, spec, pd.read_csv(spec.test_path))
    assert report.ok, report.problems
    # на медленной машине ответ даёт облегчённая ступень лестницы бейзлайна (#176) — это тоже ответ модели
    assert re.search(r"Сохранён ответ «baseline(_light)?»", proc.stderr), proc.stderr[-3000:]
