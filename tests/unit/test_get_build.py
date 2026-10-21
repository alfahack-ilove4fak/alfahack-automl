"""scripts/get_build.py: выбор свежей автосборки (без сети)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from get_build import latest_builds  # noqa: E402


def artifact(name, created, expired=False):
    return {"name": name, "created_at": created, "expired": expired}


def test_newest_first_and_only_dev_builds():
    arts = [
        artifact("submission-dev-aaa1111", "2026-10-21T09:00:00Z"),
        artifact("uv-lock", "2026-10-21T12:00:00Z"),
        artifact("submission-dev-bbb2222", "2026-10-21T10:00:00Z"),
    ]
    assert [a["name"] for a in latest_builds(arts)] == ["submission-dev-bbb2222", "submission-dev-aaa1111"]


def test_expired_builds_are_skipped():
    arts = [artifact("submission-dev-old0000", "2026-10-14T09:00:00Z", expired=True)]
    assert latest_builds(arts) == []


def test_limit():
    arts = [artifact(f"submission-dev-{i:07d}", f"2026-10-21T{i:02d}:00:00Z") for i in range(10)]
    assert len(latest_builds(arts, limit=3)) == 3
