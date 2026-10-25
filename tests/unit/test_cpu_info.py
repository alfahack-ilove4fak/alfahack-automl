"""Строка про ядра в логе (#276): видно, сколько CPU реально у агента, а не у хоста."""

from pathlib import Path

import solution


def test_cpu_info_has_core_count():
    info = solution.cpu_info()
    assert info.startswith("cpu_count=")


def test_cpu_info_survives_missing_cgroup(monkeypatch):
    def no_file(self, *args, **kwargs):
        raise FileNotFoundError(self)

    monkeypatch.setattr(Path, "read_text", no_file)
    info = solution.cpu_info()
    assert info.startswith("cpu_count=") and "cpu.max" not in info


def test_cpu_info_survives_any_read_error(monkeypatch):
    """Не только OSError: файл cgroup не в UTF-8 (UnicodeDecodeError) или любая другая ошибка — не падение агента."""

    def broken(self, *args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(Path, "read_text", broken)
    monkeypatch.setattr(solution.os, "sched_getaffinity", lambda pid: (_ for _ in ()).throw(ValueError("boom")))
    info = solution.cpu_info()
    assert info.startswith("cpu_count=") and "cpu.max" not in info and "affinity" not in info


def test_cpu_info_shows_cgroup_quota(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: "800000 100000\n")
    assert "/sys/fs/cgroup/cpu.max=800000 100000" in solution.cpu_info()


def test_cpu_info_without_affinity(monkeypatch):
    monkeypatch.delattr(solution.os, "sched_getaffinity", raising=False)  # macOS/Windows
    assert "affinity" not in solution.cpu_info()

