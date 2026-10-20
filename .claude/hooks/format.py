"""Хук Claude Code (PostToolUse на Edit|Write): форматирует изменённый .py-файл ruff'ом.

Зачем: у всех одинаковый стиль, в PR нет «шумных» правок форматирования.
Никогда не блокирует работу: при любой проблеме тихо выходит с кодом 0.
"""

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        event = json.load(sys.stdin)
        path = Path(event.get("tool_input", {}).get("file_path", ""))
        if path.suffix != ".py" or not path.exists() or "reference" in path.parts:
            return 0
        for args in (["format", "--quiet"], ["check", "--fix", "--select", "I", "--quiet"]):
            subprocess.run([sys.executable, "-m", "ruff", *args, str(path)], capture_output=True, timeout=20)
    except Exception:  # noqa: BLE001 — хук не должен мешать работе
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
