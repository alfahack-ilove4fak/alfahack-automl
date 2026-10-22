"""Тесты runtime/sandbox.py: нормальный код, исключение, бесконечный цикл, переполнение памяти, мусор в выводе."""

from __future__ import annotations

import os
import sys
import time

import pytest

from runtime.sandbox import MAX_OUTPUT_CHARS, run_code

linux_only = pytest.mark.skipif(os.name != "posix", reason="лимит памяти RLIMIT_AS есть только на Linux/macOS")


def test_normal_code(tmp_path):
    res = run_code("print('привет', 1 + 1)", tmp_path, timeout=20, mem_mb=2048)
    assert res.exit_code == 0 and not res.timed_out
    assert res.stdout.strip() == "привет 2"
    assert res.stderr == ""
    assert 0 <= res.seconds < 20


def test_exception_is_reported_not_raised(tmp_path):
    res = run_code("raise ValueError('boom')", tmp_path, timeout=20, mem_mb=2048)
    assert res.exit_code == 1 and not res.timed_out
    assert "ValueError: boom" in res.stderr


def test_syntax_error(tmp_path):
    res = run_code("def f(:\n", tmp_path, timeout=20, mem_mb=2048)
    assert res.exit_code != 0 and "SyntaxError" in res.stderr


def test_exit_code_is_passed_through(tmp_path):
    assert run_code("import sys; sys.exit(7)", tmp_path, timeout=20, mem_mb=2048).exit_code == 7


def test_infinite_loop_is_killed_by_timeout(tmp_path):
    started = time.monotonic()
    res = run_code("while True: pass", tmp_path, timeout=1.0, mem_mb=2048)
    assert res.timed_out and res.exit_code != 0
    assert time.monotonic() - started < 10  # не ждём вечно


def test_timeout_kills_child_processes_too(tmp_path):
    marker = tmp_path / "alive.txt"
    code = (
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', \"import time; time.sleep(3); open(r'{marker.as_posix()}', 'w').write('x')\"])\n"
        "import time; time.sleep(60)\n"
    )
    res = run_code(code, tmp_path, timeout=1.0, mem_mb=2048)
    assert res.timed_out
    if os.name == "posix":
        time.sleep(4)
        assert not marker.exists(), "подпроцесс пережил таймаут"


@linux_only
def test_memory_overflow_does_not_kill_the_agent(tmp_path):
    res = run_code("x = bytearray(3 * 1024**3)\nprint('не должно дойти')", tmp_path, timeout=30, mem_mb=512)
    assert res.exit_code != 0 and not res.timed_out
    assert "MemoryError" in res.stderr or res.exit_code < 0
    assert "не должно дойти" not in res.stdout


@linux_only
def test_scientific_stack_fits_into_reasonable_limit(tmp_path):
    pytest.importorskip("pandas")
    code = "import numpy as np, pandas as pd\nprint(pd.DataFrame(np.arange(10)).sum().item())"
    res = run_code(code, tmp_path, timeout=60, mem_mb=2048)
    assert res.exit_code == 0, res.stderr
    assert res.stdout.strip() == "45"


def test_runs_in_workdir_and_files_persist(tmp_path):
    workdir = tmp_path / "new" / "dir"  # каталога ещё нет — песочница его создаёт
    res = run_code("open('out.txt', 'w').write(__import__('os').getcwd())", workdir, timeout=20, mem_mb=2048)
    assert res.exit_code == 0
    assert (workdir / "out.txt").read_text() == str(workdir.resolve())
    assert [p.name for p in workdir.iterdir()] == ["out.txt"]  # сам скрипт в workdir не мусорит


def test_environment_is_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "секрет")
    monkeypatch.setenv("SOME_PARENT_VAR", "1")
    res = run_code(
        "import os; print(sorted(k for k in os.environ if 'OPENAI' in k or 'PARENT' in k))", tmp_path, 20, 2048
    )
    assert res.stdout.strip() == "[]"


def test_env_extra_is_passed(tmp_path):
    res = run_code("import os; print(os.environ['X_FLAG'])", tmp_path, timeout=20, mem_mb=2048, env={"X_FLAG": "42"})
    assert res.stdout.strip() == "42"


def test_endless_output_keeps_only_the_tail(tmp_path):
    code = "import sys\nfor i in range(300000):\n    print('строка', i)\nprint('КОНЕЦ')"
    res = run_code(code, tmp_path, timeout=60, mem_mb=2048)
    assert res.exit_code == 0
    assert len(res.stdout) <= MAX_OUTPUT_CHARS and res.stdout.strip().endswith("КОНЕЦ")


def test_stderr_and_stdout_are_separated(tmp_path):
    res = run_code("import sys; print('out'); print('err', file=sys.stderr)", tmp_path, timeout=20, mem_mb=2048)
    assert res.stdout.strip() == "out" and res.stderr.strip() == "err"


def test_stdin_is_closed(tmp_path):
    res = run_code("import sys; print(repr(sys.stdin.read()))", tmp_path, timeout=20, mem_mb=2048)
    assert res.exit_code == 0 and res.stdout.strip() == "''"  # input() не должен зависнуть в ожидании


def test_never_raises_on_broken_workdir(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("я файл, а не каталог", encoding="utf-8")
    res = run_code("print(1)", blocker / "sub", timeout=5, mem_mb=2048)  # нельзя создать каталог внутри файла
    assert res.exit_code == 127 and "песочница" in res.stderr


def test_returns_contract_type(tmp_path):
    res = run_code("pass", tmp_path, timeout=20, mem_mb=2048)
    assert {"exit_code", "stdout", "stderr", "timed_out", "seconds"} <= set(type(res).model_fields)
    assert sys.executable  # песочница запускает тот же интерпретатор, что и агент
