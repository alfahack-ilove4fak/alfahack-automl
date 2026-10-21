"""Бот слияний (ADR 0009): назначает ревьюеров и сливает готовые PR в dev. Запускается из .github/workflows/integrate.yml.

    uv run python scripts/integrate.py status                 # таблица: что бот сделает с каждым PR (ничего не меняет)
    uv run python scripts/integrate.py sync --dry-run         # то же, но с шагами «назначить / подтянуть / слить»
    uv run python scripts/integrate.py sync                   # по-настоящему (так работает workflow раз в 30 минут)
    uv run python scripts/integrate.py sync --pr 12 --comment-only   # только ревьюеры, метки и комментарий для одного PR

Правила (CONTRIBUTING.md, раздел 2): на PR метка `ready` (или `hotfix`), есть одобрение коллеги (два — если PR трогает
interfaces/, .github/, .claude/, .githooks/ или solution.py), CI зелёный, конфликтов нет, ветка не отстала от dev.
Метка `wait` останавливает бота. За один запуск сливается не больше одного PR: остальным нужно подтянуть свежий dev.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

from team_config import REPO

from tasks import team_members  # scripts/tasks.py: состав команды из docs/team.md

# Кого просим в ревьюеры по умолчанию (одобрить может любой коллега — это подсказка)
DEFAULT_REVIEWER_ROLES = {
    "agent": ["ml"],
    "ml": ["research"],
    "research": ["ml"],
    "qa": ["agent"],
    "captain": ["qa"],
}
PROTECTED_PREFIXES = ("interfaces/", ".github/", ".claude/", ".githooks/")
PROTECTED_FILES = ("solution.py",)
MARKER = "<!-- integrate-bot -->"
CHECK_PASS = {"SUCCESS", "SKIPPED", "NEUTRAL"}
LABEL_TWO = "needs-2-approvals"
EMOJI = {"merge": "✅", "update": "🔄", "wait": "⏳", "blocked": "⛔", "skip": "·"}

PR_FIELDS = (
    "number,title,url,isDraft,baseRefName,headRefName,headRefOid,author,labels,"
    "latestReviews,reviewRequests,statusCheckRollup,mergeable,files"
)


@dataclass(frozen=True)
class Decision:
    action: str  # merge | update | wait | blocked | skip
    reason: str


# ------------------------------------------------------------------ чистая логика (покрыта тестами)


def roles_by_nick() -> dict[str, str]:
    """ник (в нижнем регистре) → роль; порядок как в docs/team.md."""
    out: dict[str, str] = {}
    for role, nicks in team_members().items():
        for nick in nicks:
            out.setdefault(nick.lower(), role)
    return out


def is_protected(files: list[str]) -> bool:
    return any(f.startswith(PROTECTED_PREFIXES) or f in PROTECTED_FILES for f in files)


def ci_state(rollup: list[dict] | None) -> tuple[str, list[str]]:
    """none | pending | failed | ok и имена проверок, из-за которых не ok."""
    if not rollup:
        return "none", []
    pending, failed = [], []
    for check in rollup:
        name = check.get("name") or check.get("context") or "?"
        if check.get("__typename") == "StatusContext":
            state = check.get("state")
            if state == "SUCCESS":
                continue
            (pending if state in ("PENDING", "EXPECTED") else failed).append(name)
        elif check.get("status") != "COMPLETED":
            pending.append(name)
        elif check.get("conclusion") not in CHECK_PASS:
            failed.append(name)
    if failed:
        return "failed", failed
    return ("pending", pending) if pending else ("ok", [])


def _login(item: dict | None) -> str:
    return ((item or {}).get("login") or "").lower()


def decide(pr: dict, roles: dict[str, str], behind_by: Callable[[], int]) -> Decision:
    """Что делать с PR. behind_by вызывается, только если всё остальное в порядке (это запрос к API)."""
    author = _login(pr.get("author"))
    labels = {label["name"] for label in pr.get("labels", [])}
    files = [f["path"] for f in pr.get("files", [])]

    if pr["baseRefName"] != "dev":
        return Decision("skip", "PR не в dev: релиз в main делает дежурный дня")
    if pr.get("isDraft"):
        return Decision("skip", "черновик")
    if pr["headRefName"].startswith("team/"):
        return Decision("blocked", "ветка `team/*` не сливается в dev (CONTRIBUTING.md, раздел 1)")
    if author not in roles:
        return Decision("skip", "автор не из docs/team.md")

    need = 2 if is_protected(files) else 1
    if "wait" in labels:
        return Decision("wait", "стоит метка `wait`")
    if not labels & {"ready", "hotfix"}:
        return Decision("wait", "жду метку `ready` от автора")

    approvers, changes = set(), set()
    for review in pr.get("latestReviews", []):
        who = _login(review.get("author"))
        if who == author or who not in roles:
            continue
        if review.get("state") == "APPROVED":
            approvers.add(who)
        elif review.get("state") == "CHANGES_REQUESTED":
            changes.add(who)
    if changes:
        return Decision("blocked", f"{', '.join('@' + c for c in sorted(changes))} просит правки")
    if len(approvers) < need:
        zone = (
            " (затронуты `interfaces/`, `.github/`, `.claude/`, `.githooks/` или `solution.py`: нужно два)"
            if need == 2
            else ""
        )
        return Decision("wait", f"жду одобрений коллег: {len(approvers)} из {need}{zone}")

    if pr.get("mergeable") == "CONFLICTING":
        return Decision("blocked", "конфликт с dev: `git fetch origin && git merge origin/dev`, исправь и запушь")

    state, names = ci_state(pr.get("statusCheckRollup"))
    if state == "none":
        return Decision("wait", "CI ещё не запустился")
    if state == "pending":
        return Decision("wait", f"жду CI: {', '.join(names)}")
    if state == "failed":
        return Decision("blocked", f"CI красный: {', '.join(names)}")

    behind = behind_by()
    if behind > 0:
        return Decision("update", f"ветка отстала от dev на {behind} коммит(ов): подтягиваю и перезапускаю CI")
    return Decision("merge", "одобрения есть, CI зелёный, конфликтов нет")


def reviewers_to_request(pr: dict, roles: dict[str, str]) -> list[str]:
    """Кого ещё попросить в ревьюеры: сначала по таблице, потом остальные; нужно 1 (или 2 для зон)."""
    author = _login(pr.get("author"))
    author_role = roles.get(author)
    if not author_role:
        return []
    need = 2 if is_protected([f["path"] for f in pr.get("files", [])]) else 1
    have = {_login(r) for r in pr.get("reviewRequests", [])}
    have |= {_login(r.get("author")) for r in pr.get("latestReviews", [])}
    have.discard(author)
    table = [n for role in DEFAULT_REVIEWER_ROLES.get(author_role, []) for n, r in roles.items() if r == role]
    candidates = [n for n in table + list(roles) if n != author]
    picked: list[str] = []
    for nick in candidates:
        if len(have) + len(picked) >= need:
            break
        if nick not in have and nick not in picked:
            picked.append(nick)
    return picked


def status_text(decision: Decision) -> str:
    guide = f"https://github.com/{REPO}/blob/dev/CONTRIBUTING.md"
    return (
        f"{MARKER}\n🤖 **Бот слияний:** {EMOJI[decision.action]} {decision.reason}\n\n"
        f"<sub>Проверяю каждые 30 минут. Готов — метка `ready`, срочно — `hotfix`, подождать — `wait`. "
        f"Правила: [CONTRIBUTING.md]({guide}).</sub>"
    )


# ------------------------------------------------------------------ работа с GitHub


def gh(*args: str, check: bool = True) -> str:
    for attempt in (1, 2):
        proc = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8")
        if proc.returncode == 0 or attempt == 2:
            break
    if check and proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])}…: {(proc.stderr or proc.stdout).strip()[:300]}")
    return proc.stdout.strip()


def list_prs() -> list[dict]:
    raw = gh("pr", "list", "--repo", REPO, "--state", "open", "--limit", "100", "--json", PR_FIELDS)
    return sorted(json.loads(raw), key=lambda p: p["number"])


def behind_by_dev(pr: dict) -> int:
    return int(gh("api", f"repos/{REPO}/compare/dev...{pr['headRefOid']}", "--jq", ".behind_by"))


def post_status(number: int, decision: Decision, dry: bool) -> None:
    """Один комментарий бота на PR: создаём или правим на месте, только если текст изменился."""
    text = status_text(decision)
    raw = gh("api", f"repos/{REPO}/issues/{number}/comments", "--paginate", "--jq", ".[] | [.id, .body] | @json")
    existing = [json.loads(line) for line in raw.splitlines() if line.strip()]
    mine = next(((cid, body) for cid, body in existing if body.startswith(MARKER)), None)
    if mine and mine[1] == text:
        return
    print(f"   💬 #{number}: {decision.reason}")
    if dry:
        return
    if mine:
        gh("api", "-X", "PATCH", f"repos/{REPO}/issues/comments/{mine[0]}", "-f", f"body={text}")
    else:
        gh("api", f"repos/{REPO}/issues/{number}/comments", "-f", f"body={text}")


def ensure_reviewers(pr: dict, roles: dict[str, str], dry: bool) -> None:
    nicks = reviewers_to_request(pr, roles)
    if nicks:
        print(f"   👀 #{pr['number']}: прошу в ревьюеры {', '.join('@' + n for n in nicks)}")
        if not dry:
            gh("pr", "edit", str(pr["number"]), "--repo", REPO, "--add-reviewer", ",".join(nicks), check=False)


def ensure_zone_label(pr: dict, dry: bool) -> None:
    has = any(label["name"] == LABEL_TWO for label in pr.get("labels", []))
    want = is_protected([f["path"] for f in pr.get("files", [])])
    if has != want and not dry:
        flag = "--add-label" if want else "--remove-label"
        gh("pr", "edit", str(pr["number"]), "--repo", REPO, flag, LABEL_TWO, check=False)


def do_update(pr: dict, dry: bool) -> None:
    """Подтянуть dev в ветку PR и запустить CI вручную: пуш от токена Actions сам CI не запускает."""
    if dry:
        return
    gh(
        "api",
        "-X",
        "PUT",
        f"repos/{REPO}/pulls/{pr['number']}/update-branch",
        "-f",
        f"expected_head_sha={pr['headRefOid']}",
    )
    gh("workflow", "run", "ci.yml", "--repo", REPO, "--ref", pr["headRefName"], check=False)


def run_sync(only_pr: int | None, comment_only: bool, dry: bool) -> int:
    roles = roles_by_nick()
    merged = 0
    for pr in list_prs():
        number = pr["number"]
        if only_pr and number != only_pr:
            continue
        decision = decide(pr, roles, lambda pr=pr: behind_by_dev(pr))
        print(f"#{number} {EMOJI[decision.action]} {decision.action}: {decision.reason} — {pr['title'][:60]}")
        if decision.action == "skip":
            continue
        ensure_reviewers(pr, roles, dry)
        ensure_zone_label(pr, dry)
        if not comment_only:
            if decision.action == "merge" and merged:
                decision = Decision("update", "dev изменился после слияния другого PR: подтягиваю и перезапускаю CI")
            try:
                if decision.action == "merge":
                    print(f"   🔀 сливаю #{number}")
                    if not dry:
                        gh(
                            "pr",
                            "merge",
                            str(number),
                            "--repo",
                            REPO,
                            "--squash",
                            "--match-head-commit",
                            pr["headRefOid"],
                        )
                    merged += 1
                    decision = Decision("merge", "слит ботом (Squash)")
                elif decision.action == "update":
                    print(f"   🔄 подтягиваю dev в #{number}")
                    do_update(pr, dry)
            except RuntimeError as exc:
                decision = Decision("blocked", f"не получилось: {exc}")
        post_status(number, decision, dry)
    if merged and not dry:
        gh(
            "workflow", "run", "ci.yml", "--repo", REPO, "--ref", "dev", check=False
        )  # слияние токеном CI само не запускает
    print(f"✅ слито: {merged}" + (" (dry-run)" if dry else ""))
    return 0


def status() -> int:
    roles = roles_by_nick()
    for pr in list_prs():
        decision = decide(pr, roles, lambda pr=pr: behind_by_dev(pr))
        print(
            f"#{pr['number']:<4} {EMOJI[decision.action]} {decision.action:<8} {decision.reason} — {pr['title'][:55]}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "sync"])
    parser.add_argument("--pr", type=int, help="обработать только этот PR")
    parser.add_argument("--comment-only", action="store_true", help="только ревьюеры, метки и комментарий, без слияния")
    parser.add_argument("--dry-run", action="store_true", help="ничего не менять на GitHub")
    args = parser.parse_args()
    return status() if args.command == "status" else run_sync(args.pr, args.comment_only, args.dry_run)


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → эмодзи и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
