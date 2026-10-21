"""Однократная настройка GitHub: метки, вехи CP0–CP5, доска задач. Запускает капитан.

    uv run python scripts/bootstrap_github.py            # метки + вехи (+ доска, если есть право project)
    uv run python scripts/bootstrap_github.py --dry-run

Для доски нужно право project у gh:  gh auth refresh -s project
Повторный запуск безопасен: существующее не дублируется.
Задачи из ролевых чек-листов создаёт scripts/create_issues.py (этап 4).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from team_config import ORG as OWNER  # noqa: E402
from team_config import PROJECT_TITLE, REPO  # noqa: E402

LABELS = {
    "role:captain": ("5319e7", "Капитан / интегратор"),
    "role:agent": ("1d76db", "LLM/agent-инженер"),
    "role:ml": ("0e8a16", "ML-pipeline инженер"),
    "role:research": ("fbca04", "Исследователь / DS"),
    "role:qa": ("d93f0b", "QA / MLOps"),
    "level:L0": ("c5def5", "Разогрев"),
    "level:L1": ("bfd4f2", "Базовая"),
    "level:L2": ("7aa6f0", "Основная"),
    "level:L3": ("0052cc", "На победу"),
    "type:task": ("ededed", "Задача"),
    "type:bug": ("b60205", "Ошибка"),
    "type:hypothesis": ("c2e0c6", "Гипотеза для агента"),
    "contract": ("e99695", "Меняет interfaces/ — нужен /sync-check и ревью владельцев"),
    "handoff": ("f9d0c4", "Точка стыковки с другой ролью"),
    "harness": ("006b75", "Прогнать харнесс и показать в PR таблицу «было/стало» (workflow harness-pr)"),
}

MILESTONES = [
    ("CP0 Окружение готово", "2026-10-21T08:30:00Z", "У каждого doctor.py --role <роль> полностью ✅"),
    ("CP1 Контракты согласованы", "2026-10-21T10:00:00Z", "interfaces/ v1 одобрен всеми владельцами"),
    ("CP2 Ходячий скелет", "2026-10-21T15:30:00Z", "solution.py end-to-end, сабмит №1"),
    ("CP3 MVP", "2026-10-23T15:30:00Z", "LLM + AutoResearch, лучше простого бейзлайна"),
    ("CP4 Качество", "2026-10-25T15:30:00Z", "универсальность, ансамбли, 3-часовая репетиция"),
    ("CP5 Заморозка и сдача", "2026-10-26T15:00:00Z", "заморозка кода, финальный сабмит, питч"),
]


def gh(args: list[str], dry: bool, check: bool = True) -> str:
    print("   $ gh", " ".join(args))
    if dry:
        return ""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8")
    if check and proc.returncode != 0:
        print("     ⚠️", proc.stderr.strip()[:200])
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dry = args.dry_run

    print("1/3 Метки")
    for name, (color, desc) in LABELS.items():
        gh(["label", "create", name, "--repo", REPO, "--color", color, "--description", desc, "--force"], dry)

    print("2/3 Вехи (чек-поинты)")
    existing = (
        set()
        if dry
        else set(json.loads(gh(["api", f"repos/{REPO}/milestones?state=all", "--jq", "[.[].title]"], dry) or "[]"))
    )
    for title, due, desc in MILESTONES:
        if title in existing:
            print(f"   = {title} уже есть")
            continue
        gh(
            [
                "api",
                f"repos/{REPO}/milestones",
                "-f",
                f"title={title}",
                "-f",
                f"due_on={due}",
                "-f",
                f"description={desc}",
            ],
            dry,
        )

    print("3/3 Доска задач (GitHub Projects)")
    auth = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "project" not in auth.stdout + auth.stderr and not dry:
        print("   ⚠️ Нет права project. Выполни:  gh auth refresh -s project   и запусти скрипт снова.")
        return 0
    projects = gh(["project", "list", "--owner", OWNER, "--format", "json"], dry)
    number = None
    if projects:
        for p in json.loads(projects).get("projects", []):
            if p["title"] == PROJECT_TITLE:
                number = p["number"]
    if number is None:
        created = gh(["project", "create", "--owner", OWNER, "--title", PROJECT_TITLE, "--format", "json"], dry)
        number = json.loads(created)["number"] if created else "N"
    gh(["project", "link", str(number), "--owner", OWNER, "--repo", REPO], dry, check=False)
    print(
        f"✅ Доска: https://github.com/orgs/{OWNER}/projects/{number} — проверь PROJECT_NUMBER в scripts/team_config.py"
    )
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
