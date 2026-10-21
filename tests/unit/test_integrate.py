"""scripts/integrate.py: решения бота слияний (без сети)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from integrate import Decision, ci_state, decide, is_protected, reviewers_to_request  # noqa: E402

# порядок как в docs/team.md
ROLES = {"artem": "agent", "egor": "qa", "danil": "ml", "vasya": "captain", "roma": "research"}
GREEN = [{"__typename": "CheckRun", "name": "test", "status": "COMPLETED", "conclusion": "SUCCESS"}]


def make_pr(**over):
    pr = {
        "number": 1,
        "title": "t",
        "isDraft": False,
        "baseRefName": "dev",
        "headRefName": "ml/catboost",
        "headRefOid": "abc",
        "author": {"login": "danil"},
        "labels": [{"name": "ready"}],
        "latestReviews": [{"author": {"login": "roma"}, "state": "APPROVED"}],
        "reviewRequests": [],
        "statusCheckRollup": GREEN,
        "mergeable": "MERGEABLE",
        "files": [{"path": "ml/experiment.py"}],
    }
    pr.update(over)
    return pr


def run(pr, behind=0):
    return decide(pr, ROLES, lambda: behind)


def test_all_good_merges():
    assert run(make_pr()) == Decision("merge", "одобрения есть, CI зелёный, конфликтов нет")


def test_hotfix_label_counts_as_ready():
    assert run(make_pr(labels=[{"name": "hotfix"}])).action == "merge"


def test_no_ready_label_waits():
    assert run(make_pr(labels=[])).action == "wait"


def test_wait_label_stops_even_if_ready():
    assert run(make_pr(labels=[{"name": "ready"}, {"name": "wait"}])).action == "wait"


def test_not_dev_and_draft_and_unknown_author_are_skipped():
    assert run(make_pr(baseRefName="main")).action == "skip"
    assert run(make_pr(isDraft=True)).action == "skip"
    assert run(make_pr(author={"login": "stranger"})).action == "skip"


def test_personal_branch_is_blocked():
    assert run(make_pr(headRefName="team/danil")).action == "blocked"


def test_own_approval_and_outsider_do_not_count():
    reviews = [
        {"author": {"login": "danil"}, "state": "APPROVED"},
        {"author": {"login": "stranger"}, "state": "APPROVED"},
    ]
    assert run(make_pr(latestReviews=reviews)).action == "wait"


def test_any_teammate_approval_is_enough_for_normal_pr():
    reviews = [{"author": {"login": "egor"}, "state": "APPROVED"}]  # не «по таблице», но коллега
    assert run(make_pr(latestReviews=reviews)).action == "merge"


def test_changes_requested_blocks():
    reviews = [
        {"author": {"login": "roma"}, "state": "APPROVED"},
        {"author": {"login": "egor"}, "state": "CHANGES_REQUESTED"},
    ]
    decision = run(make_pr(latestReviews=reviews))
    assert decision.action == "blocked" and "@egor" in decision.reason


def test_protected_zone_needs_two_approvals():
    pr = make_pr(files=[{"path": "interfaces/schemas.py"}])
    assert run(pr).action == "wait"
    two = [{"author": {"login": "roma"}, "state": "APPROVED"}, {"author": {"login": "egor"}, "state": "APPROVED"}]
    assert run({**pr, "latestReviews": two}).action == "merge"


def test_conflict_blocks():
    assert run(make_pr(mergeable="CONFLICTING")).action == "blocked"


def test_ci_states():
    assert run(make_pr(statusCheckRollup=[])).action == "wait"
    pending = [{"__typename": "CheckRun", "name": "test", "status": "IN_PROGRESS", "conclusion": None}]
    assert run(make_pr(statusCheckRollup=pending)).action == "wait"
    red = [{"__typename": "CheckRun", "name": "test", "status": "COMPLETED", "conclusion": "FAILURE"}]
    decision = run(make_pr(statusCheckRollup=red))
    assert decision.action == "blocked" and "test" in decision.reason


def test_ci_state_skipped_and_status_context():
    assert ci_state([{"name": "x", "status": "COMPLETED", "conclusion": "SKIPPED"}]) == ("ok", [])
    assert ci_state([{"__typename": "StatusContext", "context": "c", "state": "PENDING"}]) == ("pending", ["c"])


def test_behind_dev_updates_branch():
    assert run(make_pr(), behind=3).action == "update"


def test_behind_is_not_asked_when_something_else_is_wrong():
    def boom():
        raise AssertionError("запрос к API не нужен")

    assert decide(make_pr(labels=[]), ROLES, boom).action == "wait"


def test_is_protected():
    assert is_protected(["solution.py"]) and is_protected([".github/workflows/ci.yml"])
    assert not is_protected(["ml/experiment.py", "docs/team.md"])


def test_reviewer_by_table_then_others():
    assert reviewers_to_request(make_pr(reviewRequests=[], latestReviews=[]), ROLES) == ["roma"]  # ml → research


def test_no_reviewer_when_someone_already_requested_or_reviewed():
    assert reviewers_to_request(make_pr(reviewRequests=[{"login": "egor"}], latestReviews=[]), ROLES) == []
    assert reviewers_to_request(make_pr(), ROLES) == []  # roma уже одобрил


def test_protected_zone_gets_two_reviewers_never_the_author():
    pr = make_pr(files=[{"path": "solution.py"}], reviewRequests=[], latestReviews=[])
    picked = reviewers_to_request(pr, ROLES)
    assert picked == ["roma", "artem"] and "danil" not in picked
