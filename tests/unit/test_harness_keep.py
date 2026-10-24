"""research/night/harness_keep.py: .env подхватывается, с --llm кэш выключен, папка прогонов не стирается (#276)."""

import os
import tempfile

import pytest

from evaluation import harness
from research.night import harness_keep


@pytest.fixture
def fake_harness(tmp_path, monkeypatch):
    """.env во временной папке и harness.main, который запоминает окружение и аргументы вместо прогона."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\nLLM_CACHE_DIR=cache-from-dotenv\n", encoding="utf-8")
    original = harness.load_env_file
    monkeypatch.setattr(harness, "load_env_file", lambda: original(env_file))
    for name in ("OPENAI_API_KEY", "LLM_CACHE_DIR"):  # setenv запоминает исходное значение: load_dotenv не утечёт
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    seen = {}

    def _main(argv):
        seen["argv"] = argv
        seen["key"] = os.environ.get("OPENAI_API_KEY")
        seen["cache"] = os.environ.get("LLM_CACHE_DIR")
        with tempfile.TemporaryDirectory(prefix="harness_") as tmp:
            seen["tmp"] = tmp
        return 0

    monkeypatch.setattr(harness, "main", _main)
    return seen


def test_llm_run_gets_key_from_dotenv_and_no_cache(fake_harness, tmp_path):
    keep = tmp_path / "runs" / "llm15"
    assert harness_keep.main([str(keep), "--task", "t/task.md", "--llm"]) == 0
    assert fake_harness["key"] == "from-dotenv"  # без этого агент пишет «LLM выключена…»
    assert fake_harness["cache"] == ""  # на платформе кэша нет
    assert fake_harness["argv"] == ["--task", "t/task.md", "--llm"]
    assert fake_harness["tmp"] == str(keep.resolve())
    assert keep.is_dir()  # временная папка подменена постоянной и не удалена


def test_cache_flag_keeps_cache_and_is_not_passed_to_harness(fake_harness, tmp_path):
    harness_keep.main([str(tmp_path / "keep"), "--llm", "--cache"])
    assert fake_harness["cache"] == "cache-from-dotenv"
    assert "--cache" not in fake_harness["argv"]


def test_shell_variables_win_over_dotenv(fake_harness, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    harness_keep.main([str(tmp_path / "keep"), "--all"])
    assert fake_harness["key"] == "from-shell"
    assert fake_harness["cache"] == "cache-from-dotenv"  # без --llm кэш не трогаем


def test_tempfile_is_restored_after_run(fake_harness, tmp_path):
    before = tempfile.TemporaryDirectory
    harness_keep.main([str(tmp_path / "keep"), "--all"])
    assert tempfile.TemporaryDirectory is before


def test_no_arguments_is_an_error_not_a_crash():
    assert harness_keep.main([]) == 2


def test_llm_without_key_stops_before_run(fake_harness, tmp_path):
    (tmp_path / ".env").unlink()  # ключа нет ни в окружении, ни в .env
    assert harness_keep.main([str(tmp_path / "keep"), "--all", "--llm"]) == 2
    assert "argv" not in fake_harness  # иначе прогон молча идёт на playbook
