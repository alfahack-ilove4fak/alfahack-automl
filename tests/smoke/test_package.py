"""Сборка ZIP и запуск «как на платформе Boosters».

Правила из инструкции платформы: в каталог задачи копируются только корневые .py-файлы и
requirements.txt, запуск — `python3 solution.py task.md` из этого каталога, submission.csv — рядом с task.md.
Если этот тест красный — сабмит на платформе получит 0 баллов.
"""

import base64
import io
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from package_submission import build  # noqa: E402

from agent.task_parser import parse_task  # noqa: E402
from evaluation.submission import validate_submission  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_zip_has_only_root_files(tmp_path):
    out = build(tmp_path / "submission.zip")
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == ["requirements.txt", "solution.py"]
        assert zf.read("requirements.txt") == (ROOT / "requirements.txt").read_bytes()


def _payload_names(zip_path: Path) -> set[str]:
    """Файлы агента, упакованные внутрь solution.py (ZIP в base64 в _PAYLOAD)."""
    with zipfile.ZipFile(zip_path) as zf:
        code = zf.read("solution.py").decode("utf-8")
    encoded = re.search(r'^_PAYLOAD = "([A-Za-z0-9+/=]*)"', code, re.MULTILINE).group(1)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as payload:
        return set(payload.namelist())


def test_prompts_are_packed(tmp_path):
    """Промпты агента (.md) едут на платформу, а инструкции CLAUDE.md — нет."""
    names = _payload_names(build(tmp_path / "submission.zip"))
    prompts = {p.relative_to(ROOT).as_posix() for p in (ROOT / "agent" / "prompts").rglob("*.md")}
    assert prompts <= names, f"нет в архиве: {sorted(prompts - names)}"
    assert not any(n.endswith("CLAUDE.md") for n in names)


@pytest.mark.skipif(
    not (ROOT / "agent" / "prompts" / "hypotheses.md").exists(),
    reason="agent/prompts ещё не в этой ветке — проверка включится сама после слияния",
)
def test_hypotheses_prompt_in_archive(tmp_path):
    assert "agent/prompts/hypotheses.md" in _payload_names(build(tmp_path / "submission.zip"))


def test_platform_like_run(tmp_path):
    out = build(tmp_path / "submission.zip")

    # Каталог задачи, как его готовит платформа: task.md + данные
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    src = ROOT / "tasks" / "public" / "regression"
    (task_dir / "task.md").write_bytes((src / "task.md").read_bytes())
    (task_dir / "data_dictionary.csv").write_bytes((src / "data_dictionary.csv").read_bytes())
    pd.read_csv(src / "train.csv", nrows=2000).to_csv(task_dir / "train.csv", index=False)
    pd.read_csv(src / "test.csv", nrows=300).to_csv(task_dir / "test.csv", index=False)
    pd.read_csv(src / "sample_submission.csv", nrows=300).to_csv(task_dir / "sample_submission.csv", index=False)

    # Платформа копирует ТОЛЬКО корневые .py и requirements.txt
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            if "/" not in name and (name.endswith(".py") or name == "requirements.txt"):
                (task_dir / name).write_bytes(zf.read(name))

    env = {**os.environ, "AGENT_TIME_BUDGET_SEC": "40", "AGENT_SAFETY_SEC": "5", "LLM_MOCK": "1"}
    env.pop("AGENT_RUN_DIR", None)
    proc = subprocess.run(
        [sys.executable, "solution.py", "task.md"],
        cwd=task_dir,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    spec = parse_task(task_dir / "task.md")
    assert spec.submission_path == task_dir / "submission.csv"
    report = validate_submission(spec.submission_path, spec, pd.read_csv(task_dir / "test.csv"))
    assert report.ok, report.problems
