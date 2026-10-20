"""Собрать ZIP для платформы: в корне solution.py + модули агента. Данные и тесты НЕ кладём.

    uv run python scripts/package_submission.py
→ dist/submission-ГГГГММДД-ЧЧММ.zip
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ["solution.py", "interfaces", "runtime", "agent", "ml", "evaluation", "research/playbook.yaml"]
SKIP = {"__pycache__", ".pytest_cache"}


def collect() -> list[Path]:
    files = []
    for item in SHIPPED:
        path = ROOT / item
        if path.is_file():
            files.append(path)
        else:
            files += [
                p
                for p in path.rglob("*")
                if p.is_file()
                and not SKIP & set(p.parts)
                and p.suffix in (".py", ".yaml", ".yml", ".json", ".txt", ".md")
                and p.name != "CLAUDE.md"
            ]
    return sorted(files)


def main() -> int:
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"submission-{time.strftime('%Y%m%d-%H%M')}.zip"
    files = collect()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out) as zf:
        assert "solution.py" in zf.namelist(), "solution.py должен лежать в корне архива"
    print(f"✅ {out.relative_to(ROOT)} — {len(files)} файлов, {out.stat().st_size / 1024:.0f} КБ")
    print("Дальше: проверка из архива (scripts/pre_submit.py, задача QA), тег на main, загрузка на платформу.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
