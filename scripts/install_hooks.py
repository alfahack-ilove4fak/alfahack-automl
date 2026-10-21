"""Подключает командные git-хуки из папки .githooks.

Запуск (из корня репозитория, любая ОС):
    uv run python scripts/install_hooks.py
Проверить:
    git config core.hooksPath      → должно вывести: .githooks
"""

import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOKS_DIR = ROOT / ".githooks"


def main() -> int:
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=ROOT, check=True)
    # На Linux/macOS хук должен быть исполняемым; на Windows бит не нужен.
    if os.name != "nt":
        for hook in HOOKS_DIR.iterdir():
            hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("✅ Хуки подключены: git config core.hooksPath = .githooks")
    print("   Теперь прямой пуш в main/dev и force push будут остановлены.")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
