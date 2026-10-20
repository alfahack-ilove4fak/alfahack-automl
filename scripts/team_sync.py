"""Обновить состав команды в документах по реальному состоянию GitHub (запускает капитан).

    uv run python scripts/team_sync.py

Источник ролей — таблица docs/team.md (её пополняет add_member.py).
Статус каждого берётся из GitHub: участник организации / приглашение ждёт ответа / нет доступа.
Обновляются: docs/team.md (колонка «Статус») и таблица ролей в docs/roles/README.md.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from team_config import ORG

ROOT = Path(__file__).resolve().parent.parent
TEAM = ROOT / "docs" / "team.md"
ROLES_README = ROOT / "docs" / "roles" / "README.md"
ROLE_ORDER = [
    ("captain", "Капитан / интегратор"),
    ("agent", "LLM/agent-инженер"),
    ("ml", "ML-pipeline инженер"),
    ("research", "Исследователь / Data Scientist"),
    ("qa", "QA / MLOps"),
]
TITLE_TO_ROLE = {title: role for role, title in ROLE_ORDER}
START, END = "<!-- team:start (генерирует scripts/team_sync.py — не править руками) -->", "<!-- team:end -->"


def gh_list(path: str) -> set[str]:
    out = subprocess.run(
        ["gh", "api", path, "--paginate", "--jq", ".[].login"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if out.returncode != 0:
        raise SystemExit(f"❌ gh api {path}: {out.stderr.strip()[:200]}")
    return set(out.stdout.split())


def read_team() -> list[dict]:
    rows = []
    for line in TEAM.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[1].startswith("@") and cells[2] in TITLE_TO_ROLE:
            rows.append({"name": cells[0], "github": cells[1][1:], "title": cells[2], "branch": cells[3]})
    return rows


def status(login: str, members: set[str], invited: set[str]) -> str:
    if login in members:
        return "✅ в организации"
    if login in invited:
        return "⏳ приглашение не принято"
    return "❌ нет доступа — проверь приглашение"


def main() -> int:
    members = gh_list(f"orgs/{ORG}/members")
    invited = gh_list(f"orgs/{ORG}/invitations")
    team = read_team()

    lines = [
        "# Состав команды",
        "",
        "Таблицу пополняет `scripts/add_member.py`, колонку «Статус» обновляет `scripts/team_sync.py` "
        "по данным GitHub. Роли и ответственность — `docs/roles/`.",
        "",
        "| Имя | GitHub | Роль | Личная ветка | Статус |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['name']} | @{r['github']} | {r['title']} | {r['branch']} | {status(r['github'], members, invited)} |"
        for r in team
    ]
    TEAM.write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_role = {role: [r for r in team if TITLE_TO_ROLE[r["title"]] == role] for role, _ in ROLE_ORDER}
    table = [START, "", "| Роль | Пакет | Материалы | Кто сейчас |", "|---|---|---|---|"]
    for role, title in ROLE_ORDER:
        people = ", ".join(f"@{r['github']}" + ("" if r["github"] in members else " ⏳") for r in by_role[role])
        table.append(
            f"| {title} | [{role}.md]({role}.md) | [resources/{role}.md](../resources/{role}.md) | "
            f"{people or '— (свободно)'} |"
        )
    free = [title for role, title in ROLE_ORDER if not by_role[role]]
    table += [
        "",
        "⏳ — приглашение в организацию ещё не принято."
        + (f" Свободные роли: {', '.join(free)} — их задачи L0–L1 раздаём на стендапе." if free else ""),
        "",
        END,
    ]
    text = ROLES_README.read_text(encoding="utf-8")
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: "\n".join(table), text, flags=re.S)
    ROLES_README.write_text(text, encoding="utf-8")

    for r in team:
        print(f"   {status(r['github'], members, invited)}  {r['name']} (@{r['github']}, {r['title']})")
    print(f"✅ Обновлены {TEAM.relative_to(ROOT)} и {ROLES_README.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
