"""Собрать ZIP для платформы Boosters (трек AutoML).

    uv run python scripts/package_submission.py
→ dist/submission-ГГГГММДД-ЧЧММ.zip

Правила платформы (инструкция «Как подготовить и загрузить решение»):
  * в каталог задачи копируются ТОЛЬКО корневые .py-файлы и requirements.txt — вложенные папки теряются;
  * запуск: `python3 solution.py task.md` в каталоге задачи; submission.csv — туда же;
  * промежуточные файлы можно создавать в каталоге задачи и в /tmp; ZIP до 100 МБ.

Поэтому собираем «самораспаковывающийся» solution.py: внутри — base64 ZIP с кодом агента
(interfaces/, runtime/, agent/ вместе с промптами, ml/, evaluation/, research/playbook.yaml,
настоящий solution.py).
При запуске он распаковывает код во временную папку и запускает его — команда разработки
работает в удобной структуре по папкам, а платформа получает ровно два корневых файла.
"""

from __future__ import annotations

import base64
import io
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_ITEMS = ["solution.py", "interfaces", "runtime", "agent", "ml", "evaluation", "research/playbook.yaml"]
PAYLOAD_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".md")  # .md — промпты агента (agent/prompts/)
SKIP_PARTS = {"__pycache__", ".pytest_cache"}
SKIP_NAMES = {"CLAUDE.md"}  # инструкции для Claude Code, агенту на платформе не нужны
MAX_ZIP_MB = 100

BOOTSTRAP = '''"""Точка входа для платформы Альфа Хак (AutoML). СОБРАНО АВТОМАТИЧЕСКИ — не править.

Сборка: scripts/package_submission.py, коммит {commit}, {built_at}.
Исходный код агента лежит ниже в _PAYLOAD (ZIP в base64). Платформа копирует только корневые
.py-файлы, поэтому код упакован внутрь: при запуске он распаковывается во временную папку
и запускается настоящий solution.py. Читаемая версия кода — в репозитории команды.
Запуск (как на платформе): python3 solution.py task.md
"""

import base64
import io
import runpy
import sys
import tempfile
import zipfile
from pathlib import Path

_PAYLOAD = "{payload}"


def _unpack() -> Path:
    target = Path(tempfile.mkdtemp(prefix="alfahack_agent_"))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(_PAYLOAD))) as archive:
        archive.extractall(target)
    return target


if __name__ == "__main__":
    source = _unpack()
    sys.path.insert(0, str(source))
    sys.argv[0] = str(source / "solution.py")
    runpy.run_path(str(source / "solution.py"), run_name="__main__")
'''


def collect_payload() -> list[Path]:
    files = []
    for item in PAYLOAD_ITEMS:
        path = ROOT / item
        if path.is_file():
            files.append(path)
            continue
        files += [
            p
            for p in path.rglob("*")
            if p.is_file()
            and not SKIP_PARTS & set(p.parts)
            and p.name not in SKIP_NAMES
            and p.suffix in PAYLOAD_SUFFIXES
        ]
    return sorted(files)


def commit_id() -> str:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return (proc.stdout.strip() or "unknown") + ("-dirty" if dirty else "")


def build(out: Path) -> Path:
    """Собрать архив по пути out и проверить, что он соответствует правилам платформы."""
    files = collect_payload()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as payload:
        for f in files:
            payload.write(f, f.relative_to(ROOT).as_posix())
    bootstrap = BOOTSTRAP.format(
        commit=commit_id(),
        built_at=time.strftime("%Y-%m-%d %H:%M"),
        payload=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("solution.py", bootstrap)
        zf.write(ROOT / "requirements.txt", "requirements.txt")

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert names == ["solution.py", "requirements.txt"], f"в корне должны быть только 2 файла, а есть: {names}"
    assert all("/" not in n for n in names), "в архиве не должно быть папок"
    size_mb = out.stat().st_size / 2**20
    assert size_mb < MAX_ZIP_MB, f"архив {size_mb:.1f} МБ больше лимита {MAX_ZIP_MB} МБ"
    return out


def main() -> int:
    out = build(ROOT / "dist" / f"submission-{time.strftime('%Y%m%d-%H%M')}.zip")
    print(f"✅ {out.relative_to(ROOT)} — solution.py + requirements.txt, {out.stat().st_size / 1024:.0f} КБ")
    print(f"   внутри solution.py упаковано файлов агента: {len(collect_payload())}, коммит {commit_id()}")
    print("Дальше: /submit-check, тег на main, загрузка на платформу.")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
