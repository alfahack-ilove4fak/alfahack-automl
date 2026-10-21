"""scripts/pre_submit.py: ловит плохие архивы и пропускает хороший.

Если проверка пропустит плохой архив — потеряем один из трёх сабмитов в день.
"""

import base64
import io
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from package_submission import build  # noqa: E402
from pre_submit import check_archive, main, run_with_limits  # noqa: E402

REQUIREMENTS = (ROOT / "requirements.txt").read_bytes()


def _fake_zip(path: Path, payload: dict[str, str], extra: dict[str, bytes] | None = None) -> Path:
    """Архив как у package_submission.py, но с подставленным кодом агента."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as inner:
        for name, code in payload.items():
            inner.writestr(name, code)
    solution = f'_PAYLOAD = "{base64.b64encode(buffer.getvalue()).decode()}"\n'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("solution.py", solution)
        zf.writestr("requirements.txt", REQUIREMENTS)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return path


GOOD_PAYLOAD = {
    "solution.py": "import os\nimport numpy as np\nfrom agent.loop import run_autoresearch\n",
    "research/playbook.yaml": "hypotheses: []\n",
}


def test_real_archive_passes(tmp_path):
    problems, _ = check_archive(build(tmp_path / "submission.zip"))
    assert problems == []


def test_extra_file_in_zip_is_caught(tmp_path):
    zip_path = _fake_zip(tmp_path / "s.zip", GOOD_PAYLOAD, extra={"agent/loop.py": b""})
    problems, _ = check_archive(zip_path)
    assert any("только solution.py и requirements.txt" in p for p in problems)


def test_foreign_import_is_caught(tmp_path):
    payload = {**GOOD_PAYLOAD, "ml/extra.py": "import requests\nfrom mcp import Client\n"}
    problems, _ = check_archive(_fake_zip(tmp_path / "s.zip", payload))
    assert any("import requests" in p for p in problems)
    assert any("import mcp" in p for p in problems)


def test_pip_and_missing_playbook_are_caught(tmp_path):
    payload = {"solution.py": "import subprocess\nsubprocess.run('pip install x', shell=True)\n"}
    problems, _ = check_archive(_fake_zip(tmp_path / "s.zip", payload))
    assert any("pip" in p for p in problems)
    assert any("research/playbook.yaml" in p for p in problems)


def test_hanging_agent_is_killed(tmp_path):
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    code, seconds, _, _ = run_with_limits(cmd, tmp_path, dict(os.environ), 1)  # Windows: без SYSTEMROOT не стартует
    assert code is None
    assert seconds < 30


def test_platform_like_run_on_public_task(tmp_path):
    report = tmp_path / "report.md"
    task = ROOT / "tasks" / "public" / "multiclass" / "task.md"
    code = main(["--task", str(task), "--minutes", "0.5", "--rows", "1500", "--report", str(report)])
    text = report.read_text(encoding="utf-8")
    assert code == 0, text
    assert "можно сабмитить" in text
