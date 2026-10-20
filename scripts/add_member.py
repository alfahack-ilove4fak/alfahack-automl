"""Подключить нового участника команды одной командой (запускает капитан).

Что делает:
  1. Отправляет приглашение в репозиторий (коллаборатор с правом записи).
  2. Даёт доступ на запись к доске задач (GitHub Projects).
  3. Создаёт личную ветку team/<имя> от dev и кладёт туда START_HERE.md
     (из шаблона docs/templates/START_HERE.md.tmpl: роль, зона, ревьюер, первая задача, стыковки).
  4. Открывает PR в dev: ник в .github/CODEOWNERS + строка в docs/team.md.

Запуск (любая ОС, из корня репозитория):
    uv run python scripts/add_member.py --name ivan --github ivan-gh --role ml
    uv run python scripts/add_member.py --name ivan --github ivan-gh --role ml --dry-run
    uv run python scripts/add_member.py --name ivan --github ivan-gh --role ml --refresh
        ↑ только обновить START_HERE.md в существующей ветке (заметки участника сохраняются)

Роли: captain, agent, ml, research, qa.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PROJECT_OWNER, PROJECT_NUMBER = "thenlaooo", 2
TEMPLATE = Path("docs/templates/START_HERE.md.tmpl")
TEAM_FILE = Path("docs/team.md")
CODEOWNERS = Path(".github/CODEOWNERS")
NOTES_HEADER = "## 5. Мои заметки"

ROLES = {
    "captain": "Капитан / интегратор",
    "agent": "LLM/agent-инженер",
    "ml": "ML-pipeline инженер",
    "research": "Исследователь / Data Scientist",
    "qa": "QA / MLOps",
}
ZONES = {
    "captain": "`solution.py`, `runtime/`, `interfaces/` (по процессу), релизы и сабмиты",
    "agent": "`agent/` — разбор задачи, LLM, гипотезы, цикл AutoResearch",
    "ml": "`ml/` — данные, модели, кросс-валидация, финальное обучение",
    "research": "`research/`, `tasks/practice/`, `docs/pitch/` — playbook, тренировочные задачи, питч",
    "qa": "`evaluation/`, `scripts/`, `tests/`, `.github/workflows/` — качество, харнесс, CI, сабмиты",
}
REVIEWER_ROLE = {"captain": "qa", "agent": "ml", "ml": "research", "research": "ml", "qa": "captain"}
PARTNERS = {
    "captain": "- **qa** — проверка перед каждым сабмитом и релизы `dev → main`.\n"
               "- **agent** — цикл AutoResearch и бюджет времени.\n- **все** — стендап, окна интеграции, контракты.",
    "agent": "- **ml** — ежедневно в 13:00: формат гипотез (`Hypothesis.params`).\n"
             "- **research** — 22.10 13:00: playbook v1 (примеры гипотез).\n"
             "- **капитан** — ключ LLM (лично), песочница для кода LLM, доли бюджета.",
    "ml": "- **agent** — ежедневно в 13:00: какие `params` ты поддерживаешь.\n"
          "- **research** — схема валидации, идеи признаков и моделей.\n- **капитан** — финальное обучение и бюджет.",
    "research": "- **agent** — 22.10 13:00: playbook v1.\n- **qa** — 22.10 18:30: тренировочные задачи.\n"
                "- **ml** — валидация и признаки.\n- **все** — каждый вечер материал для питча.",
    "qa": "- **research** — 22.10 18:30: тренировочные задачи для харнесса.\n"
          "- **капитан** — проверка перед каждым сабмитом.\n- **все** — помогаешь пройти CP0, разбираешь красный CI.",
}


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


def build_context(name: str, github: str, role: str) -> dict:
    from tasks import load_tasks, team_members  # scripts/tasks.py

    first = next((t for t in load_tasks() if t["role"] == role and t["level"] == "L0"), None)
    members = team_members()
    rev_role = REVIEWER_ROLE[role]
    rev = members.get(rev_role, [])
    reviewer = (", ".join(f"@{n}" for n in rev) + f" ({ROLES[rev_role]})" if rev
                else f"капитан @{PROJECT_OWNER} — пока в команде нет роли «{ROLES[rev_role]}»")
    return {"name": name, "github": github, "role": role, "role_title": ROLES[role], "zone": ZONES[role],
            "reviewer": reviewer, "partners": PARTNERS[role],
            "first_task": f"{first['id']} · {first['title']}" if first else "см. /next-task"}


def render_start_here(worktree: Path, ctx: dict, old: str | None = None) -> str:
    text = (worktree / TEMPLATE).read_text(encoding="utf-8").format(**ctx)
    if old and NOTES_HEADER in old:  # при обновлении сохраняем заметки участника
        text = text.split(NOTES_HEADER)[0] + NOTES_HEADER + old.split(NOTES_HEADER, 1)[1]
    return text


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


def grant_board_access(github: str, dry: bool) -> None:
    """Доступ на запись к доске (GraphQL updateProjectV2Collaborators)."""
    project_id = run(["gh", "api", "graphql", "-f",
                      f'query={{user(login:"{PROJECT_OWNER}"){{projectV2(number:{PROJECT_NUMBER}){{id}}}}}}',
                      "--jq", ".data.user.projectV2.id"], dry=dry)
    user_id = run(["gh", "api", "graphql", "-f", f'query={{user(login:"{github}"){{id}}}}',
                   "--jq", ".data.user.id"], dry=dry)
    mutation = ("mutation($p:ID!,$u:ID!){updateProjectV2Collaborators(input:{projectId:$p,"
                "collaborators:[{userId:$u,role:WRITER}]}){clientMutationId}}")
    run(["gh", "api", "graphql", "-f", f"query={mutation}", "-f", f"p={project_id}", "-f", f"u={user_id}"],
        dry=dry, check=False)


def write_personal_branch(branch: str, ctx: dict, tmp: Path, dry: bool, refresh: bool) -> None:
    wt = tmp / "personal"
    if refresh:
        run(["git", "fetch", "origin", branch], dry=dry)
        run(["git", "worktree", "add", "--detach", str(wt), f"origin/{branch}"], dry=dry)
    else:
        run(["git", "worktree", "add", "-b", branch, str(wt), "origin/dev"], dry=dry)
    if not dry:
        target = wt / "START_HERE.md"
        old = target.read_text(encoding="utf-8") if target.exists() else None
        # шаблон берём из свежего dev, а не из (возможно старой) личной ветки
        template_src = ROOT if refresh else wt
        target.write_text(render_start_here(template_src, ctx, old), encoding="utf-8")
    run(["git", "add", "START_HERE.md"], cwd=wt, dry=dry)
    run(["git", "commit", "-m", f"docs: {'обновить ' if refresh else ''}START_HERE для {ctx['name']} ({ctx['role']})"],
        cwd=wt, dry=dry, check=not refresh)
    run(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], cwd=wt, dry=dry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="имя латиницей для ветки, например ivan")
    parser.add_argument("--github", required=True, help="ник на GitHub без @")
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--dry-run", action="store_true", help="только показать команды")
    parser.add_argument("--skip-invite", action="store_true", help="не отправлять приглашение")
    parser.add_argument("--refresh", action="store_true", help="только обновить START_HERE.md в существующей ветке")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9-]{1,30}", args.name):
        raise SystemExit("❌ --name: только латиница в нижнем регистре, цифры и дефис (например, ivan или anna-k)")
    github = args.github.lstrip("@")
    ctx = build_context(args.name, github, args.role)
    dry = args.dry_run
    repo = repo_name() if not dry else "<owner>/<repo>"
    branch = f"team/{args.name}"
    run(["git", "fetch", "origin", "dev"], dry=dry)
    tmp = Path(tempfile.mkdtemp(prefix="add_member_"))

    try:
        if args.refresh:
            print(f"\n🔄 Обновляю START_HERE.md в {branch}\n")
            write_personal_branch(branch, ctx, tmp, dry, refresh=True)
            print(f"\n✅ Готово: https://github.com/{repo}/blob/{branch}/START_HERE.md")
            return 0

        print(f"\n👤 Подключаю {args.name} (@{github}) как «{ROLES[args.role]}» в {repo}\n")
        if not args.skip_invite:
            print("1/4 Приглашение в репозиторий")
            run(["gh", "api", "-X", "PUT", f"repos/{repo}/collaborators/{github}", "-f", "permission=push"], dry=dry)
        print("2/4 Доступ к доске задач")
        grant_board_access(github, dry)
        print(f"3/4 Личная ветка {branch}")
        write_personal_branch(branch, ctx, tmp, dry, refresh=False)

        print("4/4 PR в dev: CODEOWNERS + docs/team.md")
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
        pr = run(["gh", "pr", "create", "--base", "dev", "--head", pr_branch,
                  "--title", f"chore: добавить {args.name} в команду",
                  "--body", f"Новый участник: @{github}, роль `{args.role}`. Личная ветка: `{branch}`."],
                 cwd=wt2, dry=dry)
    finally:
        for sub in ("personal", "codeowners"):
            if (tmp / sub).exists():
                run(["git", "worktree", "remove", "--force", str(tmp / sub)], check=False)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n✅ Готово. Слей PR: gh pr merge {pr.rsplit('/', 1)[-1] or '<номер>'} --squash")
    print(f"Сообщение для {args.name} в чат:")
    print(f"   Прими приглашение: https://github.com/{repo}/invitations")
    print(f"   Твоя инструкция:   https://github.com/{repo}/blob/{branch}/START_HERE.md")
    print(f"   Путь на 1 час:     https://github.com/{repo}/blob/dev/ONBOARDING.md")
    print(f"   Доска задач:       https://github.com/users/{PROJECT_OWNER}/projects/{PROJECT_NUMBER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
