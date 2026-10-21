"""Тесты scripts/release.py: теги, заметки, проверки и сценарий без сети, git и gh.

Подменяем только `_run` (все внешние команды), `_now` и `_build_zip` — остальной код работает как есть.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import release  # noqa: E402

MSK = release.MSK


def _proc(out: str = "", code: int = 0, err: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, out, err)


class FakeWorld:
    """Подмена внешнего мира: отвечает на git/gh/ruff/pytest по началу команды и запоминает вызовы."""

    def __init__(self, head: str = "a" * 40, main: str = "b" * 40, tags: str = "", **flags) -> None:
        self.head, self.main, self.tags = head, main, tags
        self.flags = flags
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        c = " ".join(cmd)
        if cmd[0] == "gh":
            if "run list" in c:
                return _proc(
                    json.dumps([{"headSha": self.head, "status": "completed", "conclusion": "success", "url": "u"}])
                )
            if "pr list" in c:
                return _proc(self.flags.get("open_pr", "[]"))
            return _proc("https://example/ok")
        if cmd[0] != "git":  # ruff / pytest через sys.executable
            return _proc("ok", 1 if self.flags.get("tests_fail") and "pytest" in c else 0)
        if "status --porcelain" in c:
            return _proc(self.flags.get("dirty", ""))
        if "rev-parse origin/main" in c:
            return _proc(self.main)
        if "rev-parse origin/dev" in c:
            return _proc(self.head)
        if "rev-parse --verify HEAD^2" in c:
            return _proc("", 1)
        if "rev-parse HEAD" in c:
            return _proc(self.head)
        if "rev-list --count" in c:
            return _proc(self.flags.get("ahead", "3"))
        if "tag --list" in c:
            return _proc(self.tags)
        if "ls-remote" in c:
            return _proc("")
        if "log --no-merges" in c:
            return _proc("feat: сторож (#86)\nfix: без номера\nfeat(ml): optuna (#115)")
        return _proc("")


def _fix(monkeypatch, world: FakeWorld, now: datetime | None = None, zip_path: Path | None = None) -> None:
    monkeypatch.setattr(release, "_run", world)
    monkeypatch.setattr(release, "_now", lambda: now or datetime(2026, 10, 21, 18, 40, tzinfo=MSK))
    if zip_path is not None:
        monkeypatch.setattr(release, "_build_zip", lambda: zip_path)


# ---------------------------------------------------------------- теги


def test_next_tag_first_of_day():
    tag, number = release.next_tag(set(), datetime(2026, 10, 21, 18, 0, tzinfo=MSK))
    assert (tag, number) == ("submit-20261021-1", 1)


def test_next_tag_counts_only_today_and_takes_max():
    tags = {"submit-20261020-3", "submit-20261021-1", "submit-20261021-2"}
    assert release.next_tag(tags, datetime(2026, 10, 21, 19, 0, tzinfo=MSK))[0] == "submit-20261021-3"


def test_next_tag_uses_moscow_day_not_utc():
    # 23:30 UTC 21 октября = 02:30 МСК 22 октября: лимит платформы уже новый

    now = datetime(2026, 10, 21, 23, 30, tzinfo=UTC)
    assert release.next_tag({"submit-20261021-3"}, now) == ("submit-20261022-1", 1)


def test_existing_tags_ignores_foreign_and_reads_remote(monkeypatch):
    class W(FakeWorld):
        def __call__(self, cmd, timeout=600):
            if "ls-remote" in " ".join(cmd):
                return _proc("abc\trefs/tags/submit-20261021-2\nabc\trefs/tags/submit-bad")
            return super().__call__(cmd, timeout)

    _fix(monkeypatch, W(tags="submit-20261021-1\nv1.0\nsubmit-x"))
    assert release.existing_tags() == {"submit-20261021-1", "submit-20261021-2"}


def test_previous_tag_picks_latest_by_date(monkeypatch):
    _fix(monkeypatch, FakeWorld(tags="submit-20261020-2\nsubmit-20261021-1\nsubmit-20261020-3"))
    assert release.previous_tag("origin/main") == "submit-20261021-1"


def test_previous_tag_empty_when_no_submits(monkeypatch):
    _fix(monkeypatch, FakeWorld(tags=""))
    assert release.previous_tag("origin/main") == ""


# ---------------------------------------------------------------- проверки


def _ci(conclusion: str = "success", status: str = "completed", sha: str = "abcdef1234") -> str:
    return json.dumps([{"headSha": sha, "status": status, "conclusion": conclusion, "url": "https://ci/1"}])


def test_ci_verdict_green():
    assert release.ci_verdict("dev", "abcdef1234", _ci()).ok


def test_ci_verdict_failed_running_missing_and_garbage():
    assert not release.ci_verdict("dev", "abcdef1234", _ci(conclusion="failure")).ok
    assert "идёт" in release.ci_verdict("dev", "abcdef1234", _ci(status="in_progress")).detail
    assert "нет прогона" in release.ci_verdict("dev", "ffffffffff", _ci()).detail  # CI зелёный, но для другого коммита
    assert not release.ci_verdict("dev", "abcdef1234", "not json").ok


def test_check_limit():
    assert release.check_limit(3).ok
    assert not release.check_limit(4).ok


def test_check_clean(monkeypatch):
    _fix(monkeypatch, FakeWorld(dirty=""))
    assert release.check_clean().ok
    _fix(monkeypatch, FakeWorld(dirty=" M solution.py"))
    assert not release.check_clean().ok


def test_check_at_requires_head_equal_remote(monkeypatch):
    _fix(monkeypatch, FakeWorld(head="a" * 40, main="b" * 40))
    assert not release.check_at("main", "main").ok
    _fix(monkeypatch, FakeWorld(head="b" * 40, main="b" * 40))
    assert release.check_at("main", "main").ok


def test_print_checks_non_blocking_failure_does_not_stop(capsys):
    checks = [release.Check("a", True), release.Check("b", False, "детали", blocking=False)]
    assert release.print_checks(checks) is True
    assert release.WARN in capsys.readouterr().out
    assert release.print_checks([release.Check("c", False, "x")]) is False


# ---------------------------------------------------------------- заметки


def test_merged_prs_parses_numbers(monkeypatch):
    _fix(monkeypatch, FakeWorld())
    assert release.merged_prs("origin/main..origin/dev") == [
        ("86", "feat: сторож"),
        ("", "fix: без номера"),
        ("115", "feat(ml): optuna"),
    ]


def test_build_notes_contains_sha_and_prs(monkeypatch, tmp_path):
    _fix(monkeypatch, FakeWorld())
    zip_path = tmp_path / "x.zip"
    zip_path.write_bytes(b"zip")
    notes = release.build_notes("submit-20261021-1", "origin/main..origin/dev", "submit-20261020-2", zip_path, "c" * 40)
    assert release.sha256_of(zip_path) in notes
    assert "#86 feat: сторож" in notes and "- fix: без номера" in notes
    assert "submit-20261020-2" in notes and "ccccccc" in notes


def test_build_notes_without_previous_submit(monkeypatch):
    _fix(monkeypatch, FakeWorld())
    notes = release.build_notes("submit-20261021-1", "HEAD", "", None, "c" * 40)
    assert "предыдущих сабмитов нет" in notes


# ---------------------------------------------------------------- сценарий


def test_detect_stage(monkeypatch):
    _fix(monkeypatch, FakeWorld(head="b" * 40, main="b" * 40))
    assert release.detect_stage("auto") == "tag"
    _fix(monkeypatch, FakeWorld(head="a" * 40, main="b" * 40))
    assert release.detect_stage("auto") == "pr"
    assert release.detect_stage("tag") == "tag"


def test_dry_run_changes_nothing(monkeypatch, tmp_path, capsys):
    zip_path = tmp_path / "submission-1.zip"
    zip_path.write_bytes(b"zip")
    world = FakeWorld()
    _fix(monkeypatch, world, zip_path=zip_path)
    monkeypatch.setattr(release, "ROOT", tmp_path)

    code = release.main(["--skip-pre-submit", "--skip-tests"])

    assert code == 0
    assert "Режим проверки" in capsys.readouterr().out
    joined = [" ".join(c) for c in world.calls]
    assert not any(c.startswith("git tag -a") or "git push" in c for c in joined)
    assert not any("pr create" in c or "release create" in c for c in joined)


def test_failed_checks_stop_release_before_build(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(release, "_build_zip", lambda: built.append(1))
    _fix(monkeypatch, FakeWorld(dirty=" M x.py"))
    assert release.main(["--skip-pre-submit", "--skip-tests"]) == 1
    assert built == []


def test_over_limit_blocks_tag_unless_allowed(monkeypatch, tmp_path):
    zip_path = tmp_path / "s.zip"
    zip_path.write_bytes(b"zip")
    tags = "submit-20261021-1\nsubmit-20261021-2\nsubmit-20261021-3"
    world = FakeWorld(head="b" * 40, main="b" * 40, tags=tags)
    _fix(monkeypatch, world, zip_path=zip_path)
    monkeypatch.setattr(release, "ROOT", tmp_path)
    assert release.main(["--stage", "tag", "--skip-pre-submit", "--skip-tests"]) == 1
    # при --allow-over-limit проверка превращается в предупреждение (но без --yes тег всё равно не ставится)
    assert release.main(["--stage", "tag", "--skip-pre-submit", "--skip-tests", "--allow-over-limit"]) == 0


def test_tag_stage_with_yes_tags_pushes_and_releases(monkeypatch, tmp_path):
    zip_path = tmp_path / "submission-1.zip"
    zip_path.write_bytes(b"zip")
    world = FakeWorld(head="b" * 40, main="b" * 40)
    _fix(monkeypatch, world, zip_path=zip_path)
    monkeypatch.setattr(release, "ROOT", tmp_path)

    assert release.main(["--stage", "tag", "--yes", "--skip-pre-submit", "--skip-tests"]) == 0

    joined = [" ".join(c) for c in world.calls]
    assert any(c.startswith("git tag -a submit-20261021-1") for c in joined)
    assert any(c == "git push origin submit-20261021-1" for c in joined)
    assert any("release create submit-20261021-1" in c for c in joined)
    assert (tmp_path / "submit-20261021-1.zip").exists()  # архив назван по тегу


def test_pr_stage_does_not_duplicate_open_pr(monkeypatch, tmp_path):
    zip_path = tmp_path / "s.zip"
    zip_path.write_bytes(b"zip")
    world = FakeWorld(open_pr='[{"url": "https://github.com/x/pull/1"}]')
    _fix(monkeypatch, world, zip_path=zip_path)
    monkeypatch.setattr(release, "ROOT", tmp_path)

    assert release.main(["--stage", "pr", "--yes", "--skip-pre-submit", "--skip-tests"]) == 0
    assert not any("pr create" in " ".join(c) for c in world.calls)


def test_dry_run_flag_overrides_yes():
    assert release.parse_args(["--yes", "--dry-run"]).yes is False


def test_pr_body_warns_about_merge_commit():
    body = release.pr_body("## Релиз x\n")
    assert "merge commit" in body and "не squash" in body


@pytest.mark.parametrize("subject,expected", [("feat: x (#12)", "12"), ("feat: x", ""), ("x (#1) y", "")])
def test_pr_number_regex(subject, expected):
    m = release.PR_RE.search(subject)
    assert (m.group(1) if m else "") == expected
