"""Проверка перед сабмитом: прогоняем ZIP ровно так, как его запустит платформа.

    uv run python scripts/pre_submit.py                         # собрать zip, по 5 минут на каждую задачу tasks/public
    uv run python scripts/pre_submit.py --minutes 1 --rows 3000 # быстро: укороченные данные
    uv run python scripts/pre_submit.py --zip dist/submission-20261021-1200.zip
    uv run python scripts/pre_submit.py --task tasks/practice/x/task.md --task tasks/public/regression/task.md

Что проверяем (любой ❌ → код выхода 1, сабмитить нельзя):
  1. архив: в корне только solution.py и requirements.txt, < 100 МБ, requirements.txt совпадает с репозиторием
     и с reference/requirements.txt (набор библиотек платформы);
  2. код внутри solution.py: импорты только из стандартной библиотеки, наших пакетов и requirements.txt,
     нет запуска pip (на платформе он недоступен), нет os.rename/shutil.move (перенос файла между файловыми
     системами давал 0 баллов, #148), нет чужих библиотек внутри (.whl, .so, site-packages — только наш код);
  3. запуск: каталог задачи во временной папке (чистая среда, без кэшей прошлых запусков), туда — только
     корневые .py и requirements.txt из архива, `python solution.py task.md` из этого каталога, вне репозитория
     и без PYTHONPATH; агент обязан уложиться в бюджет, сохранить ответ бейзлайна (строка
     «Сохранён ответ «baseline»» или «…«baseline_light»» в логе) и оставить валидный submission.csv рядом с task.md.
     Какая ступень бейзлайна сработала (#176), видно в отчёте: облегчённая или «без CV» — ⚠️, ресурсов не хватило.
По умолчанию LLM выключена (LLM_MOCK=1); --llm берёт OPENAI_* из окружения и тратит общий бюджет ₽.
Отчёт — в консоль и в runs/pre_submit/report-ГГГГММДД-ЧЧММ.md.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from package_submission import MAX_ZIP_MB, build  # noqa: E402 — соседний файл scripts/

from agent.task_parser import parse_task  # noqa: E402
from evaluation.submission import validate_submission  # noqa: E402

OK, FAIL, WARN = "✅", "❌", "⚠️"
LOCAL_PACKAGES = {"interfaces", "runtime", "agent", "ml", "evaluation", "research"}
# имя при импорте → имя пакета в requirements.txt
IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "category_encoders": "category-encoders",
    "feature_engine": "feature-engine",
    "imblearn": "imbalanced-learn",
    "autogluon": "autogluon.tabular",
    "langchain_openai": "langchain-openai",
}
REQUIRED_PAYLOAD = ["solution.py", "research/playbook.yaml"]
PLATFORM_REQUIREMENTS = ROOT / "reference" / "requirements.txt"  # набор библиотек платформы от организаторов
# Перенос файла: между файловыми системами (каталог задачи ↔ /tmp) у других команд давал 0 баллов (#148)
FORBIDDEN_CALLS = {"os.rename": rb"\bos\.rename\s*\(", "shutil.move": rb"\bshutil\.move\s*\("}
# Признаки чужой библиотеки внутри архива: ставить пакеты на платформе нельзя, только requirements.txt
FOREIGN_SUFFIXES = (".whl", ".so", ".pyd", ".dll", ".egg", ".tar.gz")
FOREIGN_PARTS = ("site-packages/", ".dist-info/", ".egg-info/")
# Ответ бейзлайна сохранён: полная CV — «baseline», облегчённая ступень лестницы (solution.LIGHT_ID) — «baseline_light»
BASELINE_SAVED = re.compile(r"Сохранён ответ «(baseline(?:_light)?)» \(([^)]*)\)")
GRACE_SEC = 60  # сколько ждём сверх бюджета, прежде чем считать, что агент «завис»
# Почему цикл гипотез закончился (agent/loop.py): время вышло — штатно, гипотезы кончились — бюджет простаивает (#200)
LOOP_END = re.compile(r"(Время на исходе|Гипотезы закончились)")
UNDERUSED_BELOW = 0.5  # доля бюджета, ниже которой при длинном прогоне считаем, что агент простаивает
UNDERUSED_MIN_BUDGET_SEC = 1200  # на коротких прогонах (< 20 мин) простой не показатель


@dataclass
class TaskRun:
    name: str
    ok: bool = False
    seconds: float = 0.0
    peak_ram_mb: float = 0.0
    budget_sec: int = 0
    loop_end: str = ""  # почему закончился цикл гипотез: «время» или «гипотезы закончились»
    baseline: str = ""  # какая ступень бейзлайна сохранила ответ и с какой CV — для отчёта репетиции
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stderr_tail: str = ""


def time_usage(seconds: float, budget_sec: int, stderr: str) -> tuple[str, list[str]]:
    """(почему закончился цикл, замечания) — сколько бюджета агент реально использовал и не ушёл ли за него (#240)."""
    ends = LOOP_END.findall(stderr)
    loop_end = {"Время на исходе": "время вышло", "Гипотезы закончились": "гипотезы закончились"}.get(
        ends[-1] if ends else "", ""
    )
    warnings = []
    if budget_sec and seconds > budget_sec:
        warnings.append(f"ушёл за бюджет: {seconds:.0f} с при {budget_sec} с (платформа остановит ровно в срок)")
    elif budget_sec >= UNDERUSED_MIN_BUDGET_SEC and seconds < UNDERUSED_BELOW * budget_sec:
        warnings.append(
            f"бюджет простаивает: работал {seconds / budget_sec:.0%} ({seconds:.0f} из {budget_sec} с)"
            + (f", цикл закончился: {loop_end}" if loop_end else "")
        )
    return loop_end, warnings


# --- 1–2. Архив и код внутри него


def requirement_lines(requirements: str) -> list[str]:
    """Строки «пакет==версия» без комментариев и пустых строк — для сравнения двух requirements.txt."""
    return sorted(
        line.split("#")[0].strip().lower() for line in requirements.splitlines() if line.split("#")[0].strip()
    )


def allowed_packages(requirements: str) -> set[str]:
    names = set()
    for line in requirements.splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(line.split("==")[0].split("[")[0].strip().lower())
    return names


def read_payload(solution_code: str) -> dict[str, bytes]:
    """Достать файлы агента из самораспаковывающегося solution.py (ZIP в base64 в _PAYLOAD)."""
    match = re.search(r'^_PAYLOAD = "([A-Za-z0-9+/=]*)"', solution_code, re.MULTILINE)
    if not match:
        raise ValueError("в solution.py нет _PAYLOAD — архив собран не через package_submission.py?")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group(1)))) as payload:
        return {name: payload.read(name) for name in payload.namelist()}


def bad_imports(files: dict[str, bytes], allowed: set[str]) -> list[str]:
    """Импорты, которых не будет на платформе: не stdlib, не наш пакет и не из requirements.txt."""
    bad = []
    for name, data in sorted(files.items()):
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse(data.decode("utf-8"))
        except SyntaxError as exc:
            bad.append(f"{name}: синтаксическая ошибка ({exc.msg}, строка {exc.lineno})")
            continue
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.add(node.module.split(".")[0])
        for module in sorted(modules):
            if module in sys.stdlib_module_names or module in LOCAL_PACKAGES or module == "__future__":
                continue
            if IMPORT_TO_PACKAGE.get(module, module).lower() not in allowed:
                bad.append(f"{name}: import {module}")
    return bad


def check_archive(zip_path: Path) -> tuple[list[str], list[str]]:
    """Проверки архива и кода без запуска. Возвращает (ошибки, пояснения для отчёта)."""
    problems, notes = [], []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if sorted(names) != ["requirements.txt", "solution.py"]:
            problems.append(f"в корне должны быть только solution.py и requirements.txt, а есть: {names}")
            return problems, notes
        requirements = zf.read("requirements.txt").decode("utf-8")
        solution_code = zf.read("solution.py").decode("utf-8")
    size_mb = zip_path.stat().st_size / 2**20
    notes.append(f"размер архива {size_mb:.2f} МБ")
    if size_mb >= MAX_ZIP_MB:
        problems.append(f"архив {size_mb:.1f} МБ — больше лимита {MAX_ZIP_MB} МБ")
    if requirements != (ROOT / "requirements.txt").read_text(encoding="utf-8"):
        problems.append("requirements.txt в архиве отличается от репозитория — платформа поставит не те версии")
    if not PLATFORM_REQUIREMENTS.exists():
        problems.append(f"нет {PLATFORM_REQUIREMENTS.relative_to(ROOT)} — не с чем сверить библиотеки платформы")
    else:
        ours = set(requirement_lines(requirements))
        platform = set(requirement_lines(PLATFORM_REQUIREMENTS.read_text(encoding="utf-8")))
        if ours != platform:
            diff = sorted(ours - platform)[:3] + [f"нет {x}" for x in sorted(platform - ours)[:3]]
            problems.append(f"requirements.txt отличается от reference/requirements.txt (платформы): {diff}")

    try:
        payload = read_payload(solution_code)
    except Exception as exc:  # noqa: BLE001 — битый архив = невалидный сабмит
        problems.append(f"не удалось прочитать код агента внутри solution.py: {exc}")
        return problems, notes
    notes.append(f"файлов агента внутри solution.py: {len(payload)}")
    problems += [f"в архиве нет {name}" for name in REQUIRED_PAYLOAD if name not in payload]
    problems += [
        f"импорт вне requirements.txt — {item}" for item in bad_imports(payload, allowed_packages(requirements))
    ]
    for name, data in sorted(payload.items()):
        if name.endswith(FOREIGN_SUFFIXES) or any(part in name for part in FOREIGN_PARTS):
            problems.append(f"{name}: чужая библиотека внутри архива — на платформе только requirements.txt")
        if not name.endswith(".py"):
            continue
        if re.search(rb"pip\s+install|-m\s+pip", data):
            problems.append(f"{name}: запуск pip — на платформе недоступен")
        for call, pattern in FORBIDDEN_CALLS.items():
            if re.search(pattern, data):
                problems.append(
                    f"{name}: {call} — перенос между файловыми системами давал 0 баллов, пиши файл на месте"
                )
    return problems, notes


# --- 3. Запуск как на платформе


def prepare_task_dir(task_md: Path, zip_path: Path, dst: Path, rows: int | None) -> None:
    """Каталог задачи, как его готовит платформа: файлы задачи + корневые .py и requirements.txt из архива."""
    shutil.copytree(task_md.parent, dst, ignore=shutil.ignore_patterns("submission.csv", "__pycache__"))
    if rows:
        # Быстрый режим: в каждом CSV не больше rows строк (test и sample_submission режутся одинаково)
        for csv in dst.rglob("*.csv"):
            pd.read_csv(csv, nrows=rows).to_csv(csv, index=False)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if "/" not in name and (name.endswith(".py") or name == "requirements.txt"):
                (dst / name).write_bytes(zf.read(name))


def run_env(budget_sec: int, use_llm: bool) -> dict[str, str]:
    env = {**os.environ, "AGENT_TIME_BUDGET_SEC": str(budget_sec)}
    for key in ("PYTHONPATH", "AGENT_RUN_DIR"):  # код агента должен браться только из архива
        env.pop(key, None)
    if not use_llm:
        env["LLM_MOCK"] = "1"
    return env


def run_with_limits(
    cmd: list[str], cwd: Path, env: dict[str, str], timeout: float
) -> tuple[int | None, float, float, str]:
    """Запустить процесс и следить за ним: код выхода (None — убит по таймауту), секунды, пик RAM дерева, stderr."""
    started = time.time()
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=err)
        peak, killed = 0.0, False
        try:
            parent = psutil.Process(proc.pid)
            while proc.poll() is None:
                if time.time() - started > timeout:
                    for child in parent.children(recursive=True):
                        child.kill()
                    proc.kill()
                    proc.wait()
                    killed = True
                    break
                try:
                    tree = [parent, *parent.children(recursive=True)]
                    peak = max(peak, sum(p.memory_info().rss for p in tree) / 2**20)
                except psutil.Error:
                    pass  # процесс завершился между проверками
                time.sleep(0.5)
        except psutil.NoSuchProcess:
            proc.wait()
        seconds = time.time() - started
        err.seek(0)
        stderr = err.read().decode("utf-8", errors="replace")
    return (None if killed else proc.returncode), seconds, peak, stderr


def run_task(task_md: Path, zip_path: Path, workdir: Path, minutes: float, rows: int | None, use_llm: bool) -> TaskRun:
    name = task_md.parent.name
    result = TaskRun(name=name)
    task_dir = workdir / name
    try:
        prepare_task_dir(task_md, zip_path, task_dir, rows)
    except Exception as exc:  # noqa: BLE001 — сломанная копия задачи = проверка не прошла
        result.problems.append(f"не удалось подготовить каталог задачи: {exc}")
        return result

    budget = int(minutes * 60)
    code, result.seconds, result.peak_ram_mb, stderr = run_with_limits(
        [sys.executable, "solution.py", "task.md"], task_dir, run_env(budget, use_llm), budget + GRACE_SEC
    )
    result.stderr_tail = stderr[-3000:]
    result.budget_sec = budget
    result.loop_end, time_warnings = time_usage(result.seconds, budget, stderr)
    result.warnings += time_warnings
    if code is None:
        result.problems.append(f"не уложился в бюджет {budget} с + {GRACE_SEC} с запаса — платформа его остановит")
    elif code != 0:
        result.problems.append(f"solution.py завершился с кодом {code}")

    submission = task_dir / "submission.csv"
    if not submission.exists():
        result.problems.append("нет submission.csv рядом с task.md")
    else:
        try:
            spec = parse_task(task_dir / "task.md")
            report = validate_submission(submission, spec, pd.read_csv(spec.test_path))
            result.problems += report.problems
        except Exception as exc:  # noqa: BLE001 — не смогли проверить = не сабмитим
            result.problems.append(f"не удалось проверить ответ: {exc}")
    saved = BASELINE_SAVED.search(stderr)
    if saved is None:
        result.problems.append(
            "в логе нет «Сохранён ответ «baseline»» — бейзлайн не сохранён, сдан только тривиальный ответ"
        )
    else:
        rung, score = saved.groups()
        result.baseline = f"{rung} ({score})"
        if rung != "baseline":
            result.warnings.append("полная CV бейзлайна не успела — сохранён облегчённый, цикл гипотез пропущен")
        elif score == "без CV":
            result.warnings.append("CV бейзлайна не посчиталась ни на одной ступени — ответ модели без CV")
    result.ok = not result.problems
    return result


# --- Отчёт


def render_report(zip_path: Path, archive_problems: list[str], notes: list[str], runs: list[TaskRun], args) -> str:
    ok = not archive_problems and all(r.ok for r in runs)
    lines = [
        f"# Проверка перед сабмитом — {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Итог: {OK + ' можно сабмитить' if ok else FAIL + ' сабмитить нельзя'}",
        f"Архив: `{zip_path}` · бюджет {args.minutes} мин на задачу"
        + (f" · данные урезаны до {args.rows} строк" if args.rows else " · полные данные")
        + (" · с LLM" if args.llm else " · без LLM"),
        "",
        "## Архив и код",
        *[f"- {FAIL} {p}" for p in archive_problems],
        *[f"- {n}" for n in notes],
        *([f"- {OK} структура, размер, requirements.txt, импорты"] if not archive_problems else []),
        "",
        "## Запуск как на платформе",
        "| Задача | Итог | Время, с | Из бюджета | Цикл закончился | Пик RAM, МБ | Бейзлайн | Что не так |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        issues = "; ".join([*r.problems, *[f"{WARN} {w}" for w in r.warnings]]) or "—"
        status = OK if r.ok else FAIL
        used = f"{r.seconds / r.budget_sec:.0%} ({r.budget_sec} с)" if r.budget_sec else "—"
        lines.append(
            f"| {r.name} | {status} | {r.seconds:.0f} | {used} | {r.loop_end or '—'} | {r.peak_ram_mb:.0f} "
            f"| {r.baseline or '—'} | {issues} |"
        )
    for r in runs:
        if not r.ok:
            lines += ["", f"### Конец лога {r.name}", "```", r.stderr_tail.strip(), "```"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка ZIP перед сабмитом: запуск как на платформе")
    parser.add_argument("--zip", type=Path, help="готовый архив (по умолчанию собираем заново)")
    parser.add_argument(
        "--task", type=Path, action="append", help="task.md (можно несколько; по умолчанию tasks/public)"
    )
    parser.add_argument("--minutes", type=float, default=5, help="бюджет агента на задачу, минут (по умолчанию 5)")
    parser.add_argument("--rows", type=int, help="урезать каждый CSV до N строк (быстрая проверка формата)")
    parser.add_argument("--llm", action="store_true", help="не выключать LLM (тратит общий бюджет ₽)")
    parser.add_argument("--report", type=Path, help="куда записать отчёт (по умолчанию runs/pre_submit/)")
    args = parser.parse_args(argv)

    tasks = args.task or sorted((ROOT / "tasks" / "public").glob("*/task.md"))
    if not tasks:
        print(f"{FAIL} не нашёл ни одной задачи")
        return 1
    with tempfile.TemporaryDirectory(prefix="pre_submit_") as tmp:
        workdir = Path(tmp)
        zip_path = args.zip or build(workdir / "submission.zip")
        print(f"Архив: {zip_path}")
        archive_problems, notes = check_archive(zip_path)
        print(f"{FAIL if archive_problems else OK} архив и код")
        runs = []
        for task_md in tasks:
            print(f"… {task_md.parent.name}: запуск, бюджет {args.minutes} мин", flush=True)
            runs.append(run_task(task_md.resolve(), zip_path, workdir, args.minutes, args.rows, args.llm))
            print(f"{OK if runs[-1].ok else FAIL} {runs[-1].name}: {runs[-1].seconds:.0f} с", flush=True)
        report = render_report(args.zip or Path("(собран заново)"), archive_problems, notes, runs, args)

    out = args.report or ROOT / "runs" / "pre_submit" / f"report-{time.strftime('%Y%m%d-%H%M')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Отчёт: {out}")
    return 0 if not archive_problems and all(r.ok for r in runs) else 1


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
