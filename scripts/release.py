"""Вечерний релиз dev → main и тег сабмита (задача C-L2-3). Регламент — docs/release.md.

    uv run python scripts/release.py                    # проверки + что будет сделано (ничего не меняет в git/GitHub)
    uv run python scripts/release.py --yes              # выполнить шаг
    uv run python scripts/release.py --stage tag --yes  # шаг 2 принудительно

Релиз в два шага, потому что прямой push в main и dev запрещён (docs/decisions/0003), а слияние делает капитан:
  1. stage=pr  — мы на актуальном dev: проверки, контрольная сборка zip, PR «dev → main» с заметками.
                 Капитан сливает его кнопкой «Create a merge commit» (не squash — иначе main разойдётся с dev).
  2. stage=tag — мы на актуальном main после слияния: проверки, сборка zip, тег submit-ГГГГММДД-N
                 (дата и N — по московскому времени, N = номер сабмита за день), пуш тега, GitHub Release с zip.
По умолчанию стадия определяется по тому, на каком коммите вы стоите (HEAD == origin/main → tag, иначе pr).
Без --yes скрипт только проверяет и печатает план. Тег и PR без --yes не создаются никогда.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_config import REPO  # noqa: E402 — соседний файл scripts/

OK, FAIL, WARN = "✅", "❌", "⚠️"
MSK = timezone(timedelta(hours=3))  # у Москвы нет перехода на летнее время, tzdata на Windows не нужен
TAG_PREFIX = "submit-"
TAG_RE = re.compile(rf"^{TAG_PREFIX}(\d{{8}})-(\d+)$")
DAILY_LIMIT = 3  # лимит платформы: сабмитов в день (сброс в 00:00 МСК)
MAX_NOTES_COMMITS = 100
PR_RE = re.compile(r"\(#(\d+)\)\s*$")
PRE_SUBMIT = ROOT / "scripts" / "pre_submit.py"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    blocking: bool = True  # ❌ с blocking=False печатается как ⚠️ и релиз не останавливает


@dataclass
class Plan:
    stage: str
    tag: str
    number: int
    head: str
    notes: str = ""
    zip_path: Path | None = None
    checks: list[Check] = field(default_factory=list)


# ---------------------------------------------------------------- тонкие обёртки над внешним миром
# Тесты подменяют только _run / _now / _build_zip — остальное работает как есть, без сети и gh.


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Единственное место, где запускаются внешние команды (git, gh, ruff, pytest, сборка)."""
    try:
        return subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, "", f"команда не найдена: {cmd[0]}")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"таймаут {timeout} с: {' '.join(cmd)}")


def _now() -> datetime:
    return datetime.now(MSK)


def git(*args: str) -> str:
    """Вывод git (пустая строка при ошибке — вызывающий проверяет нужное отдельно)."""
    proc = _run(["git", *args])
    return proc.stdout.strip() if proc.returncode == 0 else ""


def gh(*args: str) -> subprocess.CompletedProcess:
    return _run(["gh", *args])


def _build_zip() -> Path:
    """Собрать архив существующим scripts/package_submission.py и вернуть путь к zip."""
    started = time.time() - 1
    proc = _run([sys.executable, str(ROOT / "scripts" / "package_submission.py")])
    if proc.returncode != 0:
        raise RuntimeError(f"package_submission.py упал:\n{proc.stdout}{proc.stderr}")
    zips = [p for p in (ROOT / "dist").glob("submission-*.zip") if p.stat().st_mtime >= started]
    if not zips:
        raise RuntimeError("package_submission.py отработал, но новый zip в dist/ не найден")
    return max(zips, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------- теги


def tag_for(date: str, number: int) -> str:
    return f"{TAG_PREFIX}{date}-{number}"


def existing_tags() -> set[str]:
    """Теги submit-* локально и на origin (чужие сабмиты видны только в remote)."""
    tags = set(git("tag", "--list", f"{TAG_PREFIX}*").split())
    remote = git("ls-remote", "--tags", "--refs", "origin", f"{TAG_PREFIX}*")
    for line in remote.splitlines():
        ref = line.split("\t")[-1]
        tags.add(ref.removeprefix("refs/tags/"))
    return {t for t in tags if TAG_RE.match(t)}


def next_tag(tags: set[str], now: datetime) -> tuple[str, int]:
    """Следующий тег за московский день: submit-ГГГГММДД-N, где N = максимальный сегодняшний + 1."""
    date = now.astimezone(MSK).strftime("%Y%m%d")
    used = [int(m.group(2)) for t in tags if (m := TAG_RE.match(t)) and m.group(1) == date]
    number = max(used, default=0) + 1
    return tag_for(date, number), number


def previous_tag(ref: str) -> str:
    """Последний тег submit-*, достижимый из ref (пусто, если сабмитов ещё не было)."""
    tags = [t for t in git("tag", "--list", f"{TAG_PREFIX}*", "--merged", ref).split() if TAG_RE.match(t)]
    return max(tags, default="")  # дата в имени — ГГГГММДД, лексикографический порядок = хронологический


# ---------------------------------------------------------------- проверки


def check_clean() -> Check:
    dirty = git("status", "--porcelain")
    return Check("рабочее дерево чистое", not dirty, "" if not dirty else "незакоммиченные файлы:\n" + dirty)


def check_at(ref: str, label: str) -> Check:
    """HEAD совпадает с origin/<ref> после fetch."""
    fetch = _run(["git", "fetch", "origin", "--tags", "--prune"])
    if fetch.returncode != 0:
        return Check(f"fetch и актуальность {label}", False, f"git fetch не удался: {fetch.stderr.strip()}")
    head, remote = git("rev-parse", "HEAD"), git("rev-parse", f"origin/{ref}")
    if head and head == remote:
        return Check(f"мы на актуальном {label}", True, remote[:7])
    hint = f"HEAD {head[:7] or '?'} ≠ origin/{ref} {remote[:7] or '?'}: переключитесь на {ref} и сделайте git pull"
    return Check(f"мы на актуальном {label}", False, hint)


def check_ci(ref: str, sha: str) -> Check:
    """Последний прогон ci на голове ветки (через gh) — зелёный и именно для этого коммита."""
    proc = gh("run", "list", "--repo", REPO, "--workflow", "ci", "--branch", ref, "--limit", "5",
              "--json", "conclusion,status,headSha,url")  # fmt: skip
    if proc.returncode != 0:
        return Check(f"CI на {ref}", False, f"gh run list не удался: {(proc.stderr or proc.stdout).strip()}")
    return ci_verdict(ref, sha, proc.stdout)


def ci_verdict(ref: str, sha: str, raw_json: str) -> Check:
    import json

    try:
        runs = [r for r in json.loads(raw_json) if r.get("headSha") == sha]
    except json.JSONDecodeError:
        return Check(f"CI на {ref}", False, "gh вернул не JSON")
    if not runs:
        return Check(f"CI на {ref}", False, f"нет прогона ci для {sha[:7]} — подождите или запустите workflow вручную")
    run = runs[0]
    if run.get("status") != "completed":
        return Check(f"CI на {ref}", False, f"прогон ещё идёт ({run.get('status')}): {run.get('url', '')}")
    if run.get("conclusion") != "success":
        return Check(f"CI на {ref}", False, f"CI {run.get('conclusion')}: {run.get('url', '')}")
    return Check(f"CI на {ref} зелёный", True, sha[:7])


def check_ruff() -> Check:
    proc = _run([sys.executable, "-m", "ruff", "check", "."])
    return Check("ruff check", proc.returncode == 0, (proc.stdout + proc.stderr).strip()[-600:])


def check_pytest() -> Check:
    proc = _run([sys.executable, "-m", "pytest", "-q", "-x"], timeout=1800)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
    return Check("pytest (все тесты)", proc.returncode == 0, tail)


def check_limit(number: int) -> Check:
    if number <= DAILY_LIMIT:
        return Check(f"сабмит №{number} за сегодня (МСК) в пределах лимита {DAILY_LIMIT}", True)
    return Check(
        f"сабмит №{number} за сегодня (МСК) — больше лимита платформы {DAILY_LIMIT}",
        False,
        "платформа не примет четвёртый сабмит до 00:00 МСК; для тега-черновика есть --allow-over-limit",
    )


def check_pre_submit(zip_path: Path, args: argparse.Namespace) -> Check | None:
    """scripts/pre_submit.py (PR #69) — только если файл уже есть в репозитории."""
    if not PRE_SUBMIT.exists() or args.skip_pre_submit:
        return None
    proc = _run(
        [sys.executable, str(PRE_SUBMIT), "--zip", str(zip_path), "--minutes", str(args.pre_submit_minutes)],
        timeout=3600,
    )
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
    return Check("pre_submit.py (прогон архива как на платформе)", proc.returncode == 0, tail)


# ---------------------------------------------------------------- заметки


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merged_prs(rev_range: str) -> list[tuple[str, str]]:
    """(номер PR или «—», заголовок) для squash-коммитов диапазона; заголовок в dev — «тип: что (#N)»."""
    out = git("log", "--no-merges", "--format=%s", rev_range)
    items = []
    for subject in out.splitlines()[:MAX_NOTES_COMMITS]:
        m = PR_RE.search(subject)
        items.append((m.group(1) if m else "", PR_RE.sub("", subject).strip()))
    return items


def build_notes(plan_tag: str, rev_range: str, base: str, zip_path: Path | None, head: str) -> str:
    lines = [f"## Релиз {plan_tag}", ""]
    lines.append(
        f"- Коммит: `{head[:7]}`" + (f", предыдущий сабмит: `{base}`" if base else ", предыдущих сабмитов нет")
    )
    if zip_path is not None:
        size_kb = zip_path.stat().st_size / 1024
        lines += [f"- Архив: `{zip_path.name}`, {size_kb:.0f} КБ", f"- sha256: `{sha256_of(zip_path)}`"]
    lines += ["", "### Слитые PR", ""]
    prs = merged_prs(rev_range)
    if not prs:
        lines.append("— нет новых коммитов —")
    for number, title in prs:
        lines.append(f"- #{number} {title}" if number else f"- {title}")
    if len(prs) >= MAX_NOTES_COMMITS:
        lines.append(f"- … показаны первые {MAX_NOTES_COMMITS}, остальное — `git log {rev_range}`")
    lines += ["", "После загрузки на платформу — строка в `docs/submissions.md` (локальный скор vs public)."]
    return "\n".join(lines) + "\n"


def pr_body(notes: str) -> str:
    return (
        "## Что сделано\nРелиз `dev → main`: всё, что слито в `dev` с прошлого сабмита.\n\n"
        "## Зачем\nСабмитим только из тега на `main` (C-L2-3, docs/release.md).\n\n"
        "## Как проверить\n```\nuv run python scripts/release.py   # на dev: те же проверки\n```\n\n"
        "## Слияние\n**Create a merge commit** (не squash!), сливает капитан. Затем на main: "
        "`git pull && uv run python scripts/release.py --yes`.\n\n"
        "## Контракты\n- [x] `interfaces/` этим PR напрямую не меняется (только через PR задач)\n\n" + notes
    )


# ---------------------------------------------------------------- сценарий


def detect_stage(requested: str) -> str:
    if requested != "auto":
        return requested
    head = git("rev-parse", "HEAD")
    return "tag" if head and head == git("rev-parse", "origin/main") else "pr"


def run_checks(stage: str, args: argparse.Namespace, number: int) -> list[Check]:
    ref = "main" if stage == "tag" else "dev"
    checks = [check_clean(), check_at(ref, ref)]
    head = git("rev-parse", "HEAD")
    if checks[-1].ok:
        checks.append(check_ci(ref, head))
        if stage == "pr" and git("rev-list", "--count", "origin/main..origin/dev") == "0":
            checks.append(Check("в dev есть что релизить", False, "origin/main уже содержит весь dev"))
        if stage == "tag":
            tagged = git("tag", "--points-at", "HEAD", "--list", f"{TAG_PREFIX}*")
            checks.append(Check("этот коммит ещё без тега сабмита", not tagged, tagged))
    checks.append(check_ruff())
    if args.skip_tests:
        checks.append(Check("pytest пропущен (--skip-tests)", True, "полагаемся на CI", blocking=False))
    else:
        checks.append(check_pytest())
    if stage == "tag":
        limit = check_limit(number)
        limit.blocking = not args.allow_over_limit
        checks.append(limit)
    return checks


def print_checks(checks: list[Check]) -> bool:
    ok = True
    for c in checks:
        mark = OK if c.ok else (FAIL if c.blocking else WARN)
        print(f"{mark} {c.name}")
        if c.detail and not c.ok:
            print("   " + c.detail.replace("\n", "\n   "))
        ok &= c.ok or not c.blocking
    return ok


def release(args: argparse.Namespace) -> int:
    stage = detect_stage(args.stage)
    tag, number = next_tag(existing_tags(), _now())
    print(f"Стадия: {stage} · тег будет {tag} (сабмит №{number} за сегодня по Москве)")
    if not args.yes:
        print("Режим проверки: ничего не создаётся и не пушится (добавьте --yes, чтобы выполнить шаг).")
    print()

    if not print_checks(run_checks(stage, args, number)):
        print(f"\n{FAIL} Проверки не пройдены — релиз остановлен.")
        return 1

    head = git("rev-parse", "HEAD")
    try:
        zip_path = _build_zip()
    except RuntimeError as exc:
        print(f"{FAIL} сборка архива: {exc}")
        return 1
    print(f"{OK} архив собран: {zip_path.name}")

    if stage == "tag":
        final = zip_path.with_name(f"{tag}.zip")  # имя по тегу — по нему архив находят потом
        shutil.copy2(zip_path, final)
        pre = check_pre_submit(final, args)
        if pre is not None and not print_checks([pre]):
            return 1
        base = previous_tag("HEAD^1" if git("rev-parse", "--verify", "HEAD^2") else "HEAD")
        rev_range = f"{base}..HEAD" if base else "HEAD"
        notes = build_notes(tag, rev_range, base, final, head)
    else:
        base = previous_tag("origin/main")
        notes = build_notes(tag, "origin/main..origin/dev", base, None, head)
        notes += (
            "\nИтоговый архив и sha256 появятся при тегировании на main (архив из dev — только контрольная сборка).\n"
        )

    notes_path = ROOT / "dist" / f"release-notes-{tag}.md"
    notes_path.parent.mkdir(exist_ok=True)
    notes_path.write_text(notes, encoding="utf-8")
    print(f"{OK} заметки: {notes_path.relative_to(ROOT).as_posix()}\n")
    print(notes)

    if not args.yes:
        print("План при --yes:")
        if stage == "pr":
            print(
                "  gh pr create --base main --head dev (PR «release: …» с этими заметками); дальше — капитан сливает merge-коммитом"
            )
        else:
            print(f"  git tag -a {tag} HEAD && git push origin {tag}; gh release create {tag} с {tag}.zip и заметками")
        return 0
    return apply_pr(args, tag, notes) if stage == "pr" else apply_tag(tag, notes, final)


def apply_pr(args: argparse.Namespace, tag: str, notes: str) -> int:
    existing = gh("pr", "list", "--repo", REPO, "--base", "main", "--head", "dev", "--state", "open", "--json", "url")
    if existing.returncode == 0 and existing.stdout.strip() not in ("", "[]"):
        print(f"{WARN} PR dev → main уже открыт: {existing.stdout.strip()} — новый не создаю")
        return 0
    cmd = ["pr", "create", "--repo", REPO, "--base", "main", "--head", "dev",
           "--title", f"release: dev → main ({tag})", "--body", pr_body(notes)]  # fmt: skip
    if args.reviewer:
        cmd += ["--reviewer", args.reviewer]
    proc = gh(*cmd)
    print((proc.stdout or proc.stderr).strip())
    return proc.returncode


def apply_tag(tag: str, notes: str, zip_path: Path) -> int:
    for cmd in (["git", "tag", "-a", tag, "-m", f"Сабмит {tag}"], ["git", "push", "origin", tag]):
        proc = _run(cmd)
        if proc.returncode != 0:
            print(f"{FAIL} {' '.join(cmd)}: {(proc.stderr or proc.stdout).strip()}")
            return 1
    print(f"{OK} тег {tag} создан и отправлен")
    rel = gh("release", "create", tag, str(zip_path), "--repo", REPO, "--title", tag, "--notes", notes)
    if rel.returncode != 0:
        print(f"{WARN} GitHub Release не создан (тег уже на месте): {(rel.stderr or rel.stdout).strip()}")
    else:
        print(rel.stdout.strip())
    print(f"Дальше: загрузить dist/{tag}.zip на платформу и записать строку в docs/submissions.md (колонка «Тег»).")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Релиз dev → main и тег сабмита", formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--yes", action="store_true", help="выполнить шаг (PR или тег); без флага — только проверки и план")
    p.add_argument("--dry-run", action="store_true", help="то же, что отсутствие --yes (оставлено для ясности)")
    p.add_argument(
        "--stage", choices=["auto", "pr", "tag"], default="auto", help="auto: HEAD == origin/main → tag, иначе pr"
    )
    p.add_argument("--skip-tests", action="store_true", help="не гонять pytest локально (CI уже зелёный)")
    p.add_argument("--skip-pre-submit", action="store_true", help="не запускать scripts/pre_submit.py")
    p.add_argument("--pre-submit-minutes", type=int, default=5, help="бюджет pre_submit.py на задачу, минут")
    p.add_argument(
        "--allow-over-limit", action="store_true", help="разрешить тег №4+ за день (платформа его не примет)"
    )
    p.add_argument("--reviewer", default="", help="ник ревьюера PR dev → main (qa, см. docs/team.md)")
    args = p.parse_args(argv)
    if args.dry_run:
        args.yes = False
    return args


def main(argv: list[str] | None = None) -> int:
    return release(parse_args(argv))


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
