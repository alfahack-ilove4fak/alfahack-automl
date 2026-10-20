"""Задачи команды из docs/roles/tasks.yaml → чек-листы в ролевых пакетах и issue на доске.

    uv run python scripts/tasks.py render              # обновить чек-листы в docs/roles/<роль>.md
    uv run python scripts/tasks.py issues --dry-run    # показать, какие issue будут созданы
    uv run python scripts/tasks.py issues              # создать недостающие issue и добавить на доску
    uv run python scripts/tasks.py board               # доложить на доску все открытые issue

Повторный запуск безопасен: issue ищутся по id задачи в заголовке и не дублируются.
Исполнитель назначается по docs/team.md (роль без человека — без исполнителя).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "docs" / "roles" / "tasks.yaml"
from team_config import BOARD_URL, CAPTAIN, PROJECT_ID_JQ, PROJECT_ID_QUERY, REPO  # noqa: E402

LEVELS = {"L0": "L0 — разогрев", "L1": "L1 — базовые", "L2": "L2 — основные", "L3": "L3 — на победу"}
MILESTONES = {
    "CP0": "CP0 Окружение готово",
    "CP1": "CP1 Контракты согласованы",
    "CP2": "CP2 Ходячий скелет",
    "CP3": "CP3 MVP",
    "CP4": "CP4 Качество",
    "CP5": "CP5 Заморозка и сдача",
}
START, END = "<!-- tasks:start (генерирует scripts/tasks.py render — не править руками) -->", "<!-- tasks:end -->"


def load_tasks() -> list[dict]:
    return yaml.safe_load(TASKS.read_text(encoding="utf-8"))["tasks"]


def team_members() -> dict[str, list[str]]:
    """роль → ники из docs/team.md (капитан — всегда)."""
    members: dict[str, list[str]] = {"captain": [CAPTAIN]}
    titles = {
        "LLM/agent-инженер": "agent",
        "ML-pipeline инженер": "ml",
        "Исследователь / Data Scientist": "research",
        "QA / MLOps": "qa",
        "Капитан / интегратор": "captain",
    }
    for line in (ROOT / "docs" / "team.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 3 and cells[1].startswith("@") and cells[2] in titles:
            nicks = members.setdefault(titles[cells[2]], [])
            if cells[1][1:] not in nicks:
                nicks.append(cells[1][1:])
    return members


# ------------------------------------------------------------------ render


def task_markdown(t: dict) -> str:
    lines = [f"#### {t['id']} · {t['title']} · {t['cp']}", "", f"*Зачем:* {t['why']}", ""]
    lines += [f"{i}. {s}" for i, s in enumerate(t["steps"], 1)]
    lines += ["", "**Готово, когда:**"] + [f"- [ ] {d}" for d in t["done"]]
    if t.get("handoff"):
        lines += ["", f"**🔗 Стыковка:** {t['handoff']}"]
    return "\n".join(lines) + "\n"


def render() -> int:
    tasks = load_tasks()
    for role in sorted({t["role"] for t in tasks}):
        path = ROOT / "docs" / "roles" / f"{role}.md"
        if not path.exists():
            print(f"⚠️ нет {path.relative_to(ROOT)} — пропускаю")
            continue
        body = [START, ""]
        for level, title in LEVELS.items():
            chunk = [t for t in tasks if t["role"] == role and t["level"] == level]
            if chunk:
                body += [f"### {title}", ""] + [task_markdown(t) for t in chunk]
        body.append(END)
        text = path.read_text(encoding="utf-8")
        block = "\n".join(body)
        new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _, b=block: b, text, flags=re.S)
        if new == text and START not in text:
            print(f"⚠️ в {path.relative_to(ROOT)} нет маркеров задач")
            continue
        path.write_text(new, encoding="utf-8")
        print(f"✅ {path.relative_to(ROOT)}: {sum(t['role'] == role for t in tasks)} задач")
    return 0


# ------------------------------------------------------------------ issues


def gh(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8")
    if check and proc.returncode != 0:
        raise SystemExit(f"❌ gh {' '.join(args[:3])}…: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


_PROJECT_ID: str | None = None


def add_to_board(issue_url: str) -> None:
    """Добавить issue на доску. Через GraphQL: `gh project item-add` в gh 2.46 молча не срабатывает."""
    global _PROJECT_ID
    if _PROJECT_ID is None:
        _PROJECT_ID = gh("api", "graphql", "-f", PROJECT_ID_QUERY, "--jq", PROJECT_ID_JQ)
    number = issue_url.rstrip("/").rsplit("/", 1)[-1]
    node_id = gh("api", f"repos/{REPO}/issues/{number}", "--jq", ".node_id")
    gh(
        "api",
        "graphql",
        "-f",
        "query=mutation($p:ID!,$c:ID!){addProjectV2ItemById(input:{projectId:$p,contentId:$c}){item{id}}}",
        "-f",
        f"p={_PROJECT_ID}",
        "-f",
        f"c={node_id}",
    )


def board() -> int:
    """Положить на доску все открытые issue, которых там ещё нет (повторный запуск безопасен)."""
    urls = json.loads(gh("issue", "list", "--repo", REPO, "--state", "open", "--limit", "500", "--json", "url"))
    for i in urls:
        add_to_board(i["url"])  # повторное добавление GitHub просто игнорирует
    print(f"✅ на доске: {len(urls)} открытых issue")
    return 0


def issue_body(t: dict) -> str:
    return (
        task_markdown(t).split("\n", 2)[2]
        + f"\n---\nРоль: `{t['role']}` · уровень {LEVELS[t['level']]} · чек-поинт {t['cp']}.\n"
        f"Полный пакет роли: `docs/roles/{t['role']}.md`. Источник задачи: `docs/roles/tasks.yaml` ({t['id']})."
    )


def issues(dry: bool) -> int:
    tasks = load_tasks()
    members = team_members()
    existing = (
        json.loads(gh("issue", "list", "--repo", REPO, "--state", "all", "--limit", "500", "--json", "title"))
        if not dry
        else []
    )
    have = {m.group(1) for i in existing if (m := re.match(r"\[([A-Z]-L\d-\d+)\]", i["title"]))}
    created = 0
    if not any(i["title"].startswith("CP0: окружение готово") for i in existing):
        print("   + CP0: окружение готово — отметься здесь")
        if not dry:
            url = gh(
                "issue",
                "create",
                "--repo",
                REPO,
                "--title",
                "CP0: окружение готово — отметься здесь",
                "--milestone",
                MILESTONES["CP0"],
                "--label",
                "type:task",
                "--body",
                "Когда `uv run python scripts/doctor.py --role <роль>` показывает «✅ ГОТОВ К РАБОТЕ», "
                "оставь здесь комментарий: роль + последняя строка вывода doctor.\n\n"
                "Капитан закрывает issue, когда отметились все. Застрял — ONBOARDING.md и чат команды.",
            )
            add_to_board(url)
    for t in tasks:
        if t["id"] in have:
            continue
        title = f"[{t['id']}] {t['title']}"
        labels = [f"role:{t['role']}", f"level:{t['level']}", "type:task"] + (["handoff"] if t.get("handoff") else [])
        args = [
            "issue",
            "create",
            "--repo",
            REPO,
            "--title",
            title,
            "--body",
            issue_body(t),
            "--milestone",
            MILESTONES[t["cp"]],
            *sum((["--label", lb] for lb in labels), []),
        ]
        for nick in members.get(t["role"], []):
            args += ["--assignee", nick]
        print(f"   + {title}  [{', '.join(members.get(t['role'], [])) or 'без исполнителя'}]")
        if dry:
            continue
        url = gh(*args)
        add_to_board(url)
        created += 1
    print(
        f"✅ создано {created}, уже было {len(have & {t['id'] for t in tasks})}"
        + (" (dry-run)" if dry else f". Доска: {BOARD_URL}")
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["render", "issues", "board"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "board":
        return board()
    return render() if args.command == "render" else issues(args.dry_run)


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
