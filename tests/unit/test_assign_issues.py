"""scripts/tasks.py: кому назначать уже созданные issue (без сети)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from tasks import pick_assignments  # noqa: E402

MEMBERS = {"qa": ["egor"], "ml": ["danil"], "agent": ["artem"], "captain": ["vasya"]}


def issue(number, role=None, assignees=(), extra_labels=()):
    labels = [{"name": f"role:{role}"}] if role else []
    labels += [{"name": name} for name in extra_labels]
    return {"number": number, "labels": labels, "assignees": [{"login": a} for a in assignees]}


def test_unassigned_role_issue_gets_the_roles_member():
    assert pick_assignments([issue(42, "qa", extra_labels=["level:L0"])], MEMBERS) == [(42, ["egor"])]


def test_already_assigned_issue_is_left_alone():
    assert pick_assignments([issue(25, "ml", assignees=["someone"])], MEMBERS) == []


def test_issue_without_role_label_is_skipped():
    assert pick_assignments([issue(7)], MEMBERS) == []


def test_role_without_people_is_skipped():
    assert pick_assignments([issue(36, "research")], MEMBERS) == []


def test_several_issues_each_to_own_role():
    result = pick_assignments([issue(1, "qa"), issue(2, "agent"), issue(3, "qa", assignees=["x"])], MEMBERS)
    assert result == [(1, ["egor"]), (2, ["artem"])]
