"""Git-хук pre-push: замена защиты веток, которой нет на GitHub Free для приватных репозиториев.

Запрещает:
  * прямой пуш в main и dev (туда — только через Pull Request);
  * удаление веток main, dev и team/*;
  * force push (перезапись истории) в main, dev и team/*.

Git передаёт на stdin строки вида:
    <local ref> <local sha> <remote ref> <remote sha>
"""

import subprocess
import sys

PROTECTED = ("refs/heads/main", "refs/heads/dev")
NO_REWRITE_PREFIXES = ("refs/heads/main", "refs/heads/dev", "refs/heads/team/")
ZERO = "0" * 40


def is_ancestor(old: str, new: str) -> bool:
    """True, если old содержится в истории new (обычный пуш, не перезапись)."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", old, new],
        capture_output=True,
    )
    return result.returncode == 0


def main() -> int:
    errors = []
    for line in sys.stdin:
        parts = line.split()
        if len(parts) != 4:
            continue
        _local_ref, local_sha, remote_ref, remote_sha = parts
        branch = remote_ref.removeprefix("refs/heads/")

        if local_sha == ZERO:
            if remote_ref.startswith(NO_REWRITE_PREFIXES):
                errors.append(f"Удалять ветку «{branch}» нельзя.")
            continue

        if remote_ref in PROTECTED:
            errors.append(
                f"Прямой пуш в «{branch}» запрещён. Создайте ветку <роль>/<задача> от dev "
                f"и откройте Pull Request (docs/git-guide.md)."
            )
            continue

        if remote_ref.startswith(NO_REWRITE_PREFIXES) and remote_sha != ZERO:
            if not is_ancestor(remote_sha, local_sha):
                errors.append(
                    f"Похоже на force push в «{branch}»: это перезапишет чужую или вашу историю. "
                    f"Сначала выполните: git pull --rebase origin {branch}"
                )

    if errors:
        print("\n⛔ pre-push остановил пуш:", file=sys.stderr)
        for err in errors:
            print(f"   • {err}", file=sys.stderr)
        print("\nЕсли вы капитан и точно знаете, что делаете, обсудите это в чате команды.\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
