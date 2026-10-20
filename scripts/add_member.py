"""Подключить нового участника команды одной командой (запускает капитан).

Что делает:
  1. Отправляет приглашение в репозиторий (коллаборатор с правом записи).
  2. Создаёт личную ветку team/<имя> от dev и кладёт туда START_HERE.md
     (из шаблона docs/templates/START_HERE.md.tmpl, если он уже есть).
  3. Открывает PR в dev: ник в .github/CODEOWNERS + строка в docs/team.md.

Запуск (любая ОС, из корня репозитория):
    uv run python scripts/add_member.py --name ivan --github ivan-gh --role ml
    uv run python scripts/add_member.py --name ivan --github ivan-gh --role ml --dry-run

Роли: captain, agent, ml, research, qa.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROLES = {
    "captain": "Капитан / интегратор",
    "agent": "LLM/agent-инженер",
    "ml": "ML-pipeline инженер",
    "research": "Исследователь / Data Scientist",
    "qa": "QA / MLOps",
}

TEMPLATE = Path("docs/templates/START_HERE.md.tmpl")
TEAM_FILE = Path("docs/team.md")
CODEOWNERS = Path(".github/CODEOWNERS")

FALLBACK_START_HERE = """# START_HERE — {name}

- **Роль:** {role_title} (`{role}`)
- **GitHub:** @{github}

Подробная инструкция появится здесь, как только капитан добавит шаблон ролевого пакета.
Пока: прочитай `README.md` в ветке `dev` и напиши капитану в чат команды.
"""


def run(cmd: list[str], cwd: Path = ROOT, dry: bool = False, check: bool = True) -> str:
    print("   $", " ".join(cmd))
    if dry:
        return ""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if check and result.returncode != 0:
        print(result.stdout, result.stderr, file=sys.stderr)
        raise SystemExit(f"❌ Команда завершилась с ошибкой: {' '.join(cmd)}")
    return result.stdout.strip()


def repo_name() -> str:
    return run(["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])


def render_start_here(worktree: Path, ctx: dict) -> str:
    tmpl = worktree / TEMPLATE
    text = tmpl.read_text(encoding="utf-8") if tmpl.exists() else FALLBACK_START_HERE
    return text.format(**ctx)


def add_codeowner(text: str, role: str, github: str) -> str:
    """Дописывает @github в строки, идущие после метки «# role:<role>» (до пустой строки)."""
    out, inside = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# role:"):
            inside = stripped == f"# role:{role}"
        elif not stripped:
            inside = False
        elif inside and not stripped.startswith("#") and f"@{github}" not in line:
            line = f"{line.rstrip()} @{github}"
        out.append(line)
    return "\n".join(out) + "\n"


def add_team_row(text: str, ctx: dict) -> str:
    if not text:
        text = (
            "# Состав команды\n\n"
            "Файл обновляет `scripts/add_member.py`. Роли и ответственность — `docs/roles/`.\n\n"
            "| Имя | GitHub | Роль | Личная ветка |\n|---|---|---|---|\n"
        )
    row = f"| {ctx['name']} | @{ctx['github']} | {ctx['role_title']} | `team/{ctx['name']}` |"
    return text if row in text else text.rstrip("\n") + "\n" + row + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="имя латиницей для ветки, например ivan")
    parser.add_argument("--github", required=True, help="ник на GitHub без @")
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--dry-run", action="store_true", help="только показать команды")
    parser.add_argument("--skip-invite", action="store_true", help="не отправлять приглашение")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9-]{1,30}", args.name):
        raise SystemExit("❌ --name: только латиница в нижнем регистре, цифры и дефис (например, ivan или anna-k)")
    github = args.github.lstrip("@")
    ctx = {"name": args.name, "github": github, "role": args.role, "role_title": ROLES[args.role]}
    dry = args.dry_run
    repo = repo_name() if not dry else "<owner>/<repo>"
    branch = f"team/{args.name}"

    print(f"\n👤 Подключаю {args.name} (@{github}) как «{ROLES[args.role]}» в {repo}\n")

    # 1. Приглашение
    if not args.skip_invite:
        print("1/3 Приглашение в репозиторий")
        run(["gh", "api", "-X", "PUT", f"repos/{repo}/collaborators/{github}", "-f", "permission=push"], dry=dry)

    run(["git", "fetch", "origin", "dev"], dry=dry)
    tmp = Path(tempfile.mkdtemp(prefix="add_member_"))
    try:
        # 2. Личная ветка
        print(f"2/3 Личная ветка {branch}")
        wt = tmp / "personal"
        run(["git", "worktree", "add", "-b", branch, str(wt), "origin/dev"], dry=dry)
        if not dry:
            (wt / "START_HERE.md").write_text(render_start_here(wt, ctx), encoding="utf-8")
        run(["git", "add", "START_HERE.md"], cwd=wt, dry=dry)
        run(["git", "commit", "-m", f"docs: START_HERE для {args.name} ({args.role})"], cwd=wt, dry=dry)
        run(["git", "push", "-u", "origin", branch], cwd=wt, dry=dry)

        # 3. PR с CODEOWNERS и составом команды
        print("3/3 PR в dev: CODEOWNERS + docs/team.md")
        pr_branch = f"captain/add-member-{args.name}"
        wt2 = tmp / "codeowners"
        run(["git", "worktree", "add", "-b", pr_branch, str(wt2), "origin/dev"], dry=dry)
        if not dry:
            co = wt2 / CODEOWNERS
            co.write_text(add_codeowner(co.read_text(encoding="utf-8"), args.role, github), encoding="utf-8")
            team = wt2 / TEAM_FILE
            team.parent.mkdir(parents=True, exist_ok=True)
            old = team.read_text(encoding="utf-8") if team.exists() else ""
            team.write_text(add_team_row(old, ctx), encoding="utf-8")
        run(["git", "add", str(CODEOWNERS), str(TEAM_FILE)], cwd=wt2, dry=dry)
        run(["git", "commit", "-m", f"chore: добавить {args.name} (@{github}) в команду"], cwd=wt2, dry=dry)
        run(["git", "push", "-u", "origin", pr_branch], cwd=wt2, dry=dry)
        run(["gh", "pr", "create", "--base", "dev", "--head", pr_branch,
             "--title", f"chore: добавить {args.name} в команду",
             "--body", f"Новый участник: @{github}, роль `{args.role}`. Личная ветка: `{branch}`."],
            cwd=wt2, dry=dry)
    finally:
        for sub in ("personal", "codeowners"):
            if (tmp / sub).exists():
                run(["git", "worktree", "remove", "--force", str(tmp / sub)], check=False)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n✅ Готово. Напиши {args.name} в чат:")
    print(f"   1) прими приглашение: https://github.com/{repo}/invitations")
    print(f"   2) открой свою инструкцию: https://github.com/{repo}/blob/{branch}/START_HERE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
