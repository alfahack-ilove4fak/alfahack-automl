"""Скачать последнюю автосборку dev (workflow build, каждые 45 минут) в папку dist/.

    uv run python scripts/get_build.py            # свежий ZIP в dist/
    uv run python scripts/get_build.py --list     # только показать последние сборки

Нужен `gh` с входом в GitHub (как для остальной работы). Сборка живёт 7 дней.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from team_config import REPO

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "submission-dev-"


def latest_builds(artifacts: list[dict], limit: int = 5) -> list[dict]:
    """Не просроченные сборки dev, новые первыми."""
    ours = [a for a in artifacts if a["name"].startswith(PREFIX) and not a.get("expired")]
    return sorted(ours, key=lambda a: a["created_at"], reverse=True)[:limit]


def gh(*args: str) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"❌ gh {' '.join(args[:3])}…: {(proc.stderr or proc.stdout).strip()[:300]}")
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="только показать последние сборки")
    args = parser.parse_args()

    raw = gh("api", f"repos/{REPO}/actions/artifacts?per_page=100")
    builds = latest_builds(json.loads(raw)["artifacts"])
    if not builds:
        print("Сборок пока нет (или все старше 7 дней). Запустить: gh workflow run build.yml -f force=true")
        return 1
    for b in builds:
        print(f"   {b['created_at']}  {b['name']}  {b['size_in_bytes'] / 1024:.0f} КБ")
    if args.list:
        return 0

    newest = builds[0]
    out = ROOT / "dist"
    out.mkdir(exist_ok=True)
    gh("run", "download", str(newest["workflow_run"]["id"]), "--repo", REPO, "-n", newest["name"], "-D", str(out))
    zips = sorted(out.glob("submission-*.zip"), key=lambda p: p.stat().st_mtime)
    print(f"✅ {newest['name']} → {zips[-1].relative_to(ROOT) if zips else out}")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → кириллица и эмодзи роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
