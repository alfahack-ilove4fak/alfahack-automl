"""scripts/check_contract_change.py: правило «изменил контракт — подними версию и запиши в журнал» (без git)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_contract_change import PROTOCOLS, SCHEMAS, check, contract_changed, version_of  # noqa: E402

OLD = '''"""Контракты."""
CONTRACT_VERSION = "1.0.0"

class Task:
    """Задача."""

    name: str
'''
LOG_OLD = "| 1.0.0 | 2026-10-20 | @x | первая версия | все |\n"


def schemas(version="1.0.0", body="    name: str\n", doc="Задача.", head="Контракты."):
    return f'"""{head}"""\nCONTRACT_VERSION = "{version}"\n\nclass Task:\n    """{doc}"""\n\n{body}'


def run(new_schemas, changelog_changed=False, changelog="", old_schemas=OLD, protocols=("x = 1\n", "x = 1\n")):
    old = {SCHEMAS: old_schemas, PROTOCOLS: protocols[0]}
    new = {SCHEMAS: new_schemas, PROTOCOLS: protocols[1]}
    return check(old, new, changelog_changed, changelog)


def test_docstring_and_comment_edits_are_not_contract_changes():
    assert not contract_changed(OLD, schemas(doc="Другое описание.", head="Другой заголовок."))
    assert not contract_changed("x = 1\n", "# комментарий\nx = 1\n")
    assert run(schemas(doc="Другое описание.")) == []


def test_new_field_without_version_bump_is_a_problem():
    problems = run(schemas(body="    name: str\n    extra: int = 0\n"))
    assert any("не повышен" in p for p in problems) and any("CHANGELOG" in p for p in problems)


def test_bump_without_changelog_is_a_problem():
    problems = run(schemas(version="1.1.0", body="    name: str\n    extra: int = 0\n"))
    assert problems == ["контракт изменён, а в interfaces/CHANGELOG.md нет новой строки"]


def test_changelog_touched_but_row_for_version_missing():
    problems = run(schemas(version="1.1.0", body="    extra: int = 0\n"), changelog_changed=True, changelog=LOG_OLD)
    assert problems == ["в interfaces/CHANGELOG.md нет строки для версии 1.1.0"]


def test_process_followed_is_ok():
    log = LOG_OLD + "| 1.1.0 | 2026-10-21 | @x | поле extra | ml |\n"
    assert run(schemas(version="1.1.0", body="    extra: int = 0\n"), changelog_changed=True, changelog=log) == []


def test_version_must_grow_not_just_change():
    log = LOG_OLD + "| 0.9.0 | 2026-10-21 | @x | откат | все |\n"
    problems = run(schemas(version="0.9.0", body="    extra: int = 0\n"), changelog_changed=True, changelog=log)
    assert any("не повышен" in p for p in problems)


def test_protocols_change_also_needs_a_bump():
    problems = run(OLD, protocols=("def f(a): ...\n", "def f(a, b): ...\n"))
    assert any("protocols.py" in p for p in problems)


def test_missing_files_are_ignored():
    assert check({SCHEMAS: None, PROTOCOLS: None}, {SCHEMAS: OLD, PROTOCOLS: "x = 1\n"}, False, "") == []


def test_version_parsing():
    assert version_of('CONTRACT_VERSION = "2.10.3"\n') == (2, 10, 3)
    assert version_of("x = 1\n") is None
