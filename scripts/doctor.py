"""Проверка готовности рабочего места. Запуск из корня репозитория:

    uv run python scripts/doctor.py --role <captain|agent|ml|research|qa>
    uv run python scripts/doctor.py --role ml --quick     # без дымовых тестов (быстро)
    uv run python scripts/doctor.py --role agent --llm    # + реальный запрос к LLM (тратит копейки)

✅ — всё хорошо · ❌ — надо исправить (команда напечатана ниже) · ⚠️ — не блокирует, но прочитай.
Работает только на стандартной библиотеке, поэтому запускается даже до установки зависимостей.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from team_config import INVITATION_URL, REPO  # noqa: E402 — соседний файл scripts/

OS = platform.system()  # Windows | Darwin | Linux
OK, FAIL, WARN = "✅", "❌", "⚠️"

ROLES = ["captain", "agent", "ml", "research", "qa"]
CLAUDE_COMMANDS = ["onboard", "next-task", "sync-check", "explain", "pre-pr"]
CLAUDE_AGENTS = ["code-reviewer", "test-writer", "ml-mentor"]
CORE_IMPORTS = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "sklearn": "1.5.2",
    "lightgbm": "4.6.0",
    "pydantic": "2.10.6",
    "openai": "1.68.2",
    "yaml": None,
    "psutil": None,
    "pytest": None,
}
ROLE_IMPORTS = {
    "ml": {"catboost": "1.2.8", "xgboost": "2.1.4", "optuna": "4.2.1"},
    "research": {"matplotlib": "3.9.4", "seaborn": "0.13.2", "ipykernel": None},
    "agent": {"openai": "1.68.2", "tenacity": "9.0.0"},
    "qa": {"ruff": None},
}
ROLE_SYNC = {
    "ml": "uv sync --group full",
    "research": "uv sync --group research",
    "agent": "uv sync",
    "qa": "uv sync",
    "captain": "uv sync",
}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str | dict[str, str] = field(default="")

    def fix_for_os(self) -> str:
        if isinstance(self.fix, dict):
            return self.fix.get(OS) or self.fix.get("*", "")
        return self.fix


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except FileNotFoundError:
        return 127, "команда не найдена"
    except subprocess.TimeoutExpired:
        return 124, f"не ответила за {timeout} с"


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


# ------------------------------------------------------------------ общие проверки


def check_python() -> Check:
    v = sys.version_info
    in_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    if (v.major, v.minor) == (3, 12) and in_venv:
        return Check("Python 3.12 из окружения проекта", OK, f"{v.major}.{v.minor}.{v.micro}")
    return Check(
        "Python 3.12 из окружения проекта",
        FAIL,
        f"запущен Python {v.major}.{v.minor} ({sys.executable})",
        "uv python install 3.12 && uv sync && uv run python scripts/doctor.py",
    )


def check_uv() -> Check:
    if not have("uv"):
        return Check(
            "uv установлен",
            FAIL,
            "uv не найден в PATH",
            {
                "Windows": 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
                "*": "curl -LsSf https://astral.sh/uv/install.sh | sh   (медленно? см. docs/install.md)",
            },
        )
    return Check("uv установлен", OK, run(["uv", "--version"])[1])


def check_git() -> list[Check]:
    if not have("git"):
        return [
            Check(
                "git установлен",
                FAIL,
                "",
                {
                    "Windows": "winget install --id Git.Git -e",
                    "Darwin": "xcode-select --install",
                    "Linux": "sudo apt install -y git",
                },
            )
        ]
    out = [Check("git установлен", OK, run(["git", "--version"])[1])]
    name, email = run(["git", "config", "user.name"])[1], run(["git", "config", "user.email"])[1]
    if name and email:
        out.append(Check("git знает, кто ты", OK, f"{name} <{email}>"))
    else:
        out.append(
            Check(
                "git знает, кто ты",
                FAIL,
                "не заданы user.name / user.email",
                'git config --global user.name "Ник" && git config --global user.email "почта"',
            )
        )
    hooks = run(["git", "config", "core.hooksPath"])[1]
    out.append(
        Check(
            "git-хук защиты веток",
            OK if hooks == ".githooks" else FAIL,
            hooks or "не подключён",
            "uv run python scripts/install_hooks.py",
        )
    )
    branch = run(["git", "branch", "--show-current"])[1]
    out.append(
        Check("текущая ветка", WARN if branch == "main" else OK, branch, "git switch dev   (в main не работаем)")
    )
    return out


def check_gh() -> list[Check]:
    if not have("gh"):
        return [
            Check(
                "GitHub CLI (gh)",
                FAIL,
                "",
                {
                    "Windows": "winget install --id GitHub.cli -e",
                    "Darwin": "brew install gh",
                    "Linux": "sudo apt install -y gh",
                },
            )
        ]
    out = [Check("GitHub CLI (gh)", OK, run(["gh", "--version"])[1].splitlines()[0])]
    code, text = run(["gh", "auth", "status"])
    out.append(
        Check(
            "вход в GitHub",
            OK if code == 0 else FAIL,
            next((ln.strip() for ln in text.splitlines() if "Logged in" in ln), text[:120]),
            "gh auth login",
        )
    )
    code, text = run(["gh", "repo", "view", REPO, "--json", "name"])
    out.append(
        Check(
            "доступ к репозиторию",
            OK if code == 0 else FAIL,
            REPO,
            f"прими приглашение в организацию: {INVITATION_URL} (или напиши капитану)",
        )
    )
    return out


def check_claude() -> list[Check]:
    if not have("claude"):
        return [
            Check(
                "Claude Code установлен",
                FAIL,
                "",
                {
                    "Windows": "irm https://claude.ai/install.ps1 | iex",
                    "*": "curl -fsSL https://claude.ai/install.sh | bash",
                },
            )
        ]
    out = [Check("Claude Code установлен", OK, run(["claude", "--version"])[1])]
    missing = ["CLAUDE.md"] if not (ROOT / "CLAUDE.md").exists() else []
    missing += [f"/{c}" for c in CLAUDE_COMMANDS if not (ROOT / ".claude" / "commands" / f"{c}.md").exists()]
    missing += [f"агент {a}" for a in CLAUDE_AGENTS if not (ROOT / ".claude" / "agents" / f"{a}.md").exists()]
    skills = list((ROOT / ".claude" / "skills").glob("*/SKILL.md"))
    if not skills:
        missing.append("скиллы .claude/skills/*/SKILL.md")
    out.append(
        Check(
            "проектные команды, агенты, скиллы Claude",
            FAIL if missing else OK,
            f"нет: {', '.join(missing)}"
            if missing
            else f"{len(CLAUDE_COMMANDS)} команд, " f"{len(CLAUDE_AGENTS)} агентов, {len(skills)} скиллов",
            "git switch dev && git pull   (настройки Claude лежат в репозитории)",
        )
    )
    out.extend(check_claude_plugins())
    return out


def check_claude_plugins() -> list[Check]:
    """Список обязательных плагинов и MCP-серверов — в .claude/team-requirements.json."""
    req_file = ROOT / ".claude" / "team-requirements.json"
    if not req_file.exists():
        return [Check("плагины и MCP Claude", FAIL, "нет .claude/team-requirements.json", "git pull")]
    req = json.loads(req_file.read_text(encoding="utf-8"))
    out = []
    code, plugins = run(["claude", "plugin", "list"], timeout=60)
    for plugin in req.get("plugins", []):
        ok = code == 0 and plugin["name"] in plugins
        out.append(Check(f"плагин {plugin['name']}", OK if ok else FAIL, plugin.get("why", ""), plugin["install"]))
    code, servers = run(["claude", "mcp", "list"], timeout=90)
    for server in req.get("mcp", []):
        ok = code == 0 and server["name"] in servers
        out.append(Check(f"MCP {server['name']}", OK if ok else FAIL, server.get("why", ""), server["fix"]))
    return out


SYSTEM_LIB_FIX = {
    "Windows": "winget install --id Microsoft.VCRedist.2015+.x64 -e   (docs/install.md, шаг 3а)",
    "Darwin": "brew install libomp   (docs/install.md, шаг 3а)",
    "Linux": "sudo apt install -y libgomp1   (docs/install.md, шаг 3а)",
}


def check_imports(packages: dict[str, str | None], title: str, fix: str) -> Check:
    problems = []
    for module, version in packages.items():
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError:
            problems.append(f"{module} не установлен")
            continue
        except Exception as exc:  # noqa: BLE001 — установлен, но не загружается (нет системной библиотеки)
            problems.append(f"{module} не загружается: {type(exc).__name__}")
            fix = SYSTEM_LIB_FIX.get(OS, fix)
            continue
        have_version = getattr(mod, "__version__", None)
        if version and have_version and have_version.split("+")[0] != version:
            problems.append(f"{module} {have_version} вместо {version}")
    return Check(title, FAIL if problems else OK, "; ".join(problems) or f"{len(packages)} пакетов", fix)


def check_data() -> Check:
    expected = {"regression": 53_308, "multiclass": 87_670}
    problems = []
    for task, rows in expected.items():
        train = ROOT / "tasks" / "public" / task / "train.csv"
        if not train.exists():
            problems.append(f"нет {train.relative_to(ROOT)}")
            continue
        with train.open("rb") as fh:
            n = sum(1 for _ in fh) - 1
        if n != rows:
            problems.append(f"{task}: {n} строк вместо {rows}")
    return Check(
        "данные публичных задач",
        FAIL if problems else OK,
        "; ".join(problems) or "2 задачи на месте",
        "git pull   (данные лежат в репозитории, tasks/public/)",
    )


def check_env(strict: bool) -> Check:
    env = ROOT / ".env"
    if not env.exists():
        return Check("файл .env", FAIL, "нет файла", {"Windows": "copy .env.example .env", "*": "cp .env.example .env"})
    values = {}
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    missing = [k for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL") if not values.get(k)]
    if missing:
        return Check(
            "ключ LLM в .env",
            FAIL if strict else WARN,
            f"пусто: {', '.join(missing)} (агент работает и без LLM, на playbook)",
            "впиши значения в .env — ключ даёт капитан лично, НЕ через чат",
        )
    return Check("ключ LLM в .env", OK, "OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL заданы (значения не показываю)")


def check_llm_call() -> Check:
    try:
        from dotenv import load_dotenv

        os.environ["LLM_CACHE_DIR"] = ""  # проверяем связь, а не кэш: load_dotenv ниже пустое значение не перезапишет
        load_dotenv(ROOT / ".env")
        sys.path.insert(0, str(ROOT))
        from agent.llm_client import LLMClient

        # Qwen рассуждает: скрытые размышления тратят max_tokens, при 10–40 ответ приходит пустым
        answer = LLMClient(timeout=60).chat([{"role": "user", "content": "Ответь одним словом: готов"}], max_tokens=300)
    except Exception as exc:  # noqa: BLE001
        return Check("запрос к LLM", FAIL, f"{type(exc).__name__}: {exc}", "проверь .env и OPENAI_BASE_URL")
    return Check(
        "запрос к LLM",
        OK if answer.strip() else FAIL,
        (answer.strip() or "нет ответа")[:60],
        "проверь значения в .env; если ключ верный — напиши капитану",
    )


def check_tests(quick: bool) -> Check:
    target = ["tests/contracts", "tests/unit"] if quick else ["tests"]
    code, text = run([sys.executable, "-m", "pytest", "-x", *target], timeout=900)  # -q уже в addopts
    lines = [ln.strip("= ") for ln in text.splitlines() if ln.strip()]
    summary = [ln for ln in lines if any(w in ln for w in ("passed", "failed", "error"))]
    tail = (summary or lines or [""])[-1][:150]
    return Check(
        "тесты" + (" (без дымовых)" if quick else " (включая дымовой прогон агента)"),
        OK if code == 0 else FAIL,
        tail,
        "uv run pytest -x   — и пришли хвост вывода в чат",
    )


# ------------------------------------------------------------------ ролевые проверки


def role_checks(role: str, args) -> list[Check]:
    out = []
    if role in ROLE_IMPORTS:
        out.append(check_imports(ROLE_IMPORTS[role], f"библиотеки роли {role}", ROLE_SYNC[role]))
    if role == "captain":
        code, text = run(["gh", "auth", "status"])
        out.append(
            Check(
                "gh: право работать с доской (scope project)",
                OK if "project" in text else FAIL,
                "",
                "gh auth refresh -s project",
            )
        )
        code, text = run(["gh", "api", f"repos/{REPO}", "--jq", ".permissions.admin"])
        out.append(Check("права администратора репозитория", OK if text == "true" else FAIL, text, "только владелец"))
    if role == "agent":
        out.append(check_env(strict=True))
        if args.llm:
            out.append(check_llm_call())
    if role == "qa":
        code, _ = run(["gh", "run", "list", "--repo", REPO, "--limit", "1"])
        out.append(Check("доступ к GitHub Actions", OK if code == 0 else FAIL, "", "gh auth login"))
        out.append(
            Check(
                "Docker (для эмуляции платформы)",
                OK if have("docker") else WARN,
                "есть" if have("docker") else "нет — не обязателен, есть Codespaces",
                "https://docs.docker.com/get-docker/",
            )
        )
    return out


# ------------------------------------------------------------------ вывод


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → значки ✅ и кириллица ломают вывод
        stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=ROLES, help="твоя роль — добавит ролевые проверки")
    parser.add_argument("--quick", action="store_true", help="не запускать дымовой прогон агента")
    parser.add_argument("--llm", action="store_true", help="сделать реальный запрос к LLM")
    args = parser.parse_args()

    sections: list[tuple[str, list[Check]]] = [
        ("Инструменты", [check_python(), check_uv(), *check_git(), *check_gh()]),
        ("Claude Code", check_claude()),
        (
            "Проект",
            [
                check_imports(CORE_IMPORTS, "библиотеки ядра", "uv sync"),
                check_data(),
                check_env(strict=False),
                check_tests(args.quick),
            ],
        ),
    ]
    if args.role:
        sections.append((f"Роль: {args.role}", role_checks(args.role, args)))

    print(f"\n🩺 doctor — {OS}, Python {platform.python_version()}, папка {ROOT}\n")
    fails = warns = 0
    for title, checks in sections:
        print(f"── {title}")
        for c in checks:
            print(f"  {c.status} {c.name}" + (f" — {c.detail}" if c.detail else ""))
            if c.status != OK and c.fix_for_os():
                print(f"       ↳ {c.fix_for_os()}")
            fails += c.status == FAIL
            warns += c.status == WARN
        print()
    if not args.role:
        print(f"{WARN} Роль не указана — добавь --role <{'|'.join(ROLES)}> для ролевых проверок.\n")
    if fails:
        print(f"{FAIL} Исправь пункты с ❌ ({fails} шт.) и запусти doctor снова. Застрял на 15+ минут — пиши в чат.")
        return 1
    print(
        f"{OK} ГОТОВ К РАБОТЕ"
        + (f" (предупреждений: {warns})" if warns else "")
        + ". Следующий шаг — /next-task в Claude Code."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
