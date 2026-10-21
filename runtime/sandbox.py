"""Песочница для кода от LLM (задача C-L2-1, ADR 0008 D4).

Код запускается в отдельном процессе Python, поэтому бесконечный цикл, переполнение памяти или исключение
не роняют агента: во всех случаях возвращается `CodeResult`, исключений наружу не летит.

    from runtime.sandbox import run_code
    res = run_code("print(1 + 1)", workdir=Path("runs/latest/sandbox"), timeout=60, mem_mb=4096)
    res.exit_code, res.stdout, res.stderr, res.timed_out, res.seconds

Что гарантируется:
  - таймаут по реальному времени: по истечении убивается весь процесс (на Linux — вся группа процессов);
  - лимит памяти `RLIMIT_AS` (только Linux/macOS): код получает `MemoryError` или умирает, агент жив.
    Это лимит ВИРТУАЛЬНОЙ памяти: `import pandas, lightgbm` в процессе требует ≈ 1,5 ГБ, ставьте `mem_mb` ≥ 2048;
  - чистое окружение: ключи (`OPENAI_*`) и прочие переменные родителя в процесс не попадают;
  - stdout/stderr читаются «хвостом» (последние `MAX_OUTPUT_CHARS` символов), бесконечный `print` не съедает память агента.

Чего НЕТ: это не защита от злонамеренного кода — файловая система и сеть не изолированы (на платформе сети нет,
а код пишет наша же LLM). На Windows лимита памяти нет, только таймаут (платформа — Linux, так что это лишь для отладки).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from interfaces.schemas import CodeResult

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

MAX_OUTPUT_CHARS = 12000  # как в reference/baseline.py: хвост вывода, остальное отбрасываем
KILL_GRACE_SEC = 2.0  # сколько ждём завершения процесса после kill, прежде чем сдаться
_ENV_PASSTHROUGH = ("LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT")  # нужны интерпретатору на некоторых ОС


def _clean_env(workdir: Path, extra: dict[str, str] | None) -> dict[str, str]:
    env = {
        "PATH": os.defpath,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(workdir),
        "MPLBACKEND": "Agg",
    }
    for key in _ENV_PASSTHROUGH:
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(extra or {})
    return env


def _limit_memory(mem_mb: int):
    """preexec_fn: лимит виртуальной памяти и никаких core-файлов. Выполняется в дочернем процессе до exec."""

    def apply() -> None:
        limit = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _drain(stream, sink: deque[str]) -> None:
    """Читать поток до конца, храня только последние символы (иначе `while True: print()` съест память)."""
    total = 0
    for chunk in iter(lambda: stream.read(4096), ""):
        sink.append(chunk)
        total += len(chunk)
        while total - len(sink[0]) >= MAX_OUTPUT_CHARS and len(sink) > 1:
            total -= len(sink.popleft())


def _tail(sink: deque[str]) -> str:
    return "".join(sink)[-MAX_OUTPUT_CHARS:]


def _kill(proc: subprocess.Popen) -> None:
    """Убить процесс и (на POSIX) всю его группу — код мог запустить подпроцессы."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, 9)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_code(
    code: str,
    workdir: Path,
    timeout: float,
    mem_mb: int,
    env: dict[str, str] | None = None,
) -> CodeResult:
    """Выполнить `code` в отдельном процессе `python -I`; рабочий каталог — `workdir` (создаётся при отсутствии).

    timeout — секунд реального времени; mem_mb — лимит виртуальной памяти процесса (Linux), МБ;
    env — дополнительные переменные окружения (по умолчанию окружение пустое, см. `_clean_env`).
    Никогда не бросает исключений: сбои запуска возвращаются как `exit_code=127` и текст в `stderr`.
    """
    started = time.monotonic()
    out: deque[str] = deque()
    err: deque[str] = deque()
    try:
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="sandbox-"
        ) as script_dir:  # скрипт — вне workdir, чтобы не мусорить в нём
            script = Path(script_dir) / "main.py"
            script.write_text(code, encoding="utf-8")
            kwargs: dict = {}
            if os.name == "posix":
                kwargs["start_new_session"] = True  # своя группа процессов → можно убить всё дерево
                if resource is not None and mem_mb > 0:
                    kwargs["preexec_fn"] = _limit_memory(mem_mb)
            proc = subprocess.Popen(
                [sys.executable, "-I", "-u", str(script)],
                cwd=workdir,
                env=_clean_env(workdir, env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **kwargs,
            )
            readers = [
                threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
                threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True),
            ]
            for t in readers:
                t.start()
            timed_out = False
            try:
                proc.wait(timeout=max(timeout, 0.1))
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(proc)
                try:
                    proc.wait(timeout=KILL_GRACE_SEC)
                except subprocess.TimeoutExpired:
                    pass
            for t in readers:
                t.join(timeout=KILL_GRACE_SEC)
            code_ = proc.returncode if proc.returncode is not None else -9
            return CodeResult(
                exit_code=code_,
                stdout=_tail(out),
                stderr=_tail(err),
                timed_out=timed_out,
                seconds=round(time.monotonic() - started, 3),
            )
    except Exception as exc:  # песочница не имеет права ронять агента
        return CodeResult(
            exit_code=127,
            stdout=_tail(out),
            stderr=f"песочница не смогла запустить код: {type(exc).__name__}: {exc}",
            timed_out=False,
            seconds=round(time.monotonic() - started, 3),
        )
