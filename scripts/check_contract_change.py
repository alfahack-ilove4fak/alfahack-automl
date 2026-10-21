"""Проверка процесса изменения контрактов (ADR 0009, CONTRIBUTING.md раздел 5). Работает на стандартной библиотеке.

    python3 scripts/check_contract_change.py --base origin/dev

Правило: если в PR по смыслу изменились interfaces/schemas.py или interfaces/protocols.py (комментарии и docstring не в счёт),
то CONTRACT_VERSION в schemas.py должен вырасти, а в interfaces/CHANGELOG.md появиться строка этой версии.
Заменяет прежний контроль капитана: «забыл поднять версию» ловит CI, а не человек. Запускается в ci.yml на каждый PR.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys

SCHEMAS = "interfaces/schemas.py"
PROTOCOLS = "interfaces/protocols.py"
CHANGELOG = "interfaces/CHANGELOG.md"
VERSION_RE = re.compile(r'^CONTRACT_VERSION\s*=\s*"(\d+)\.(\d+)\.(\d+)"', re.M)


def normalized(source: str) -> str:
    """Дерево кода без docstring: правка комментариев и docstring контрактом не считается."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def contract_changed(old: str, new: str) -> bool:
    return normalized(old) != normalized(new)


def version_of(source: str) -> tuple[int, int, int] | None:
    match = VERSION_RE.search(source)
    return tuple(int(x) for x in match.groups()) if match else None  # type: ignore[return-value]


def fmt(version: tuple[int, int, int] | None) -> str:
    return ".".join(map(str, version)) if version else "?"


def check(
    old: dict[str, str | None], new: dict[str, str | None], changelog_changed: bool, changelog_text: str
) -> list[str]:
    """Список проблем (пустой — всё в порядке). old/new: путь → содержимое файла (None — файла нет)."""
    changed = [
        name
        for name in (SCHEMAS, PROTOCOLS)
        if old.get(name) is not None and new.get(name) is not None and contract_changed(old[name], new[name])
    ]
    if not changed:
        return []
    problems = []
    old_v, new_v = version_of(old[SCHEMAS] or ""), version_of(new[SCHEMAS] or "")
    if old_v is None or new_v is None:
        problems.append('не нашёл строку `CONTRACT_VERSION = "X.Y.Z"` в interfaces/schemas.py')
    elif new_v <= old_v:
        problems.append(
            f"контракт изменён ({', '.join(changed)}), а CONTRACT_VERSION не повышен ({fmt(old_v)} → {fmt(new_v)}): "
            "patch — новое необязательное поле, minor — новый контракт или функция, major — переименование или удаление"
        )
    if not changelog_changed:
        problems.append(f"контракт изменён, а в {CHANGELOG} нет новой строки")
    elif new_v and f"| {fmt(new_v)} |" not in changelog_text:
        problems.append(f"в {CHANGELOG} нет строки для версии {fmt(new_v)}")
    return problems


def git(*args: str) -> str | None:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8")
    return proc.stdout if proc.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="origin/dev", help="с чем сравнивать (ветка, в которую идёт PR)")
    args = parser.parse_args()

    fork = (git("merge-base", args.base, "HEAD") or "").strip()
    if not fork:
        print(f"⚠️ не нашёл общего предка с {args.base} (git fetch origin?) — проверку пропускаю")
        return 0
    changed_files = set((git("diff", "--name-only", f"{fork}..HEAD") or "").split())
    old = {name: git("show", f"{fork}:{name}") for name in (SCHEMAS, PROTOCOLS)}
    new = {name: git("show", f"HEAD:{name}") for name in (SCHEMAS, PROTOCOLS)}
    changelog = git("show", f"HEAD:{CHANGELOG}") or ""
    problems = check(old, new, CHANGELOG in changed_files, changelog)
    if problems:
        print("❌ Контракты изменены не по процессу (CONTRIBUTING.md, раздел 5):")
        for problem in problems:
            print(f"   - {problem}")
        print(
            "Что сделать: подними CONTRACT_VERSION в interfaces/schemas.py и добавь строку в interfaces/CHANGELOG.md."
        )
        return 1
    print("✅ Контракты: изменений нет или процесс соблюдён.")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → кириллица и эмодзи роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
