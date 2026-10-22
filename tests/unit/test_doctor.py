"""doctor: «вход в GitHub» не даёт ложный ❌ при обрыве сети (#103)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import doctor  # noqa: E402

LOGGED_IN = (0, "github.com\n  ✓ Logged in to github.com account someone (keyring)")
NETWORK_GLITCH = (
    1,
    "github.com\n  X Failed to log in to github.com account someone (keyring)\n  - The token is invalid",
)
NOT_LOGGED = (1, "You are not logged into any GitHub hosts. To log in, run: gh auth login")


@pytest.fixture
def fake_gh(monkeypatch):
    """Подменяем gh: auth — по очереди из списка ответов, repo view — заданный код."""
    calls = {"auth": 0}

    def setup(auth_answers, repo_code):
        def run(cmd, timeout=60):
            if cmd[:2] == ["gh", "--version"]:
                return 0, "gh version 2.46.0"
            if cmd[:3] == ["gh", "auth", "status"]:
                answer = auth_answers[min(calls["auth"], len(auth_answers) - 1)]
                calls["auth"] += 1
                return answer
            if cmd[:3] == ["gh", "repo", "view"]:
                return repo_code, ""
            raise AssertionError(cmd)

        monkeypatch.setattr(doctor, "have", lambda binary: True)
        monkeypatch.setattr(doctor, "run", run)
        monkeypatch.setattr(doctor.time, "sleep", lambda s: None)
        return calls

    return setup


def _login(checks):
    return next(c for c in checks if c.name == "вход в GitHub")


def test_logged_in(fake_gh):
    calls = fake_gh([LOGGED_IN], repo_code=0)
    assert _login(doctor.check_gh()).status == doctor.OK
    assert calls["auth"] == 1


def test_glitch_then_ok_is_green(fake_gh):
    calls = fake_gh([NETWORK_GLITCH, LOGGED_IN], repo_code=0)
    assert _login(doctor.check_gh()).status == doctor.OK
    assert calls["auth"] == 2


def test_glitch_every_time_but_repo_opens_is_warning(fake_gh):
    calls = fake_gh([NETWORK_GLITCH], repo_code=0)
    check = _login(doctor.check_gh())
    assert check.status == doctor.WARN and "нестабильна" in check.detail
    assert calls["auth"] == doctor.AUTH_ATTEMPTS


def test_really_broken_login_is_red(fake_gh):
    fake_gh([NETWORK_GLITCH], repo_code=1)
    checks = doctor.check_gh()
    assert _login(checks).status == doctor.FAIL
    assert next(c for c in checks if c.name == "доступ к репозиторию").status == doctor.FAIL


def test_not_logged_in_is_red_without_retries(fake_gh):
    calls = fake_gh([NOT_LOGGED], repo_code=1)
    assert _login(doctor.check_gh()).status == doctor.FAIL
    assert calls["auth"] == 1


# ------------------------------------------------------------------ запрос к LLM


@pytest.fixture
def fake_llm(monkeypatch):
    """Подменяем LLMClient: настоящий .env не читаем, в сеть не ходим."""
    import dotenv

    from agent.llm_client import LLMClient

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setenv("LLM_CACHE_DIR", "")

    def setup(available, answer):
        monkeypatch.setattr(LLMClient, "available", lambda self: available)
        monkeypatch.setattr(LLMClient, "chat", lambda self, *a, **kw: answer)

    return setup


def test_llm_call_none_answer_is_fail_not_crash(fake_llm):
    """chat() вернул None (LLM не ответила) — ❌ с пояснением, а не AttributeError до печати таблицы."""
    fake_llm(available=True, answer=None)
    check = doctor.check_llm_call()
    assert check.status == doctor.FAIL
    assert "нет ответа" in check.detail


def test_llm_call_without_client_says_why(fake_llm):
    """Нет ключа или LLM_MOCK=1 — говорим об этом прямо, а не отсылаем к пустому логу."""
    fake_llm(available=False, answer=None)
    check = doctor.check_llm_call()
    assert check.status == doctor.FAIL
    assert "клиент не создан" in check.detail


def test_llm_call_ok_shows_answer_and_time(fake_llm):
    fake_llm(available=True, answer="  готов\n")
    check = doctor.check_llm_call()
    assert check.status == doctor.OK
    assert check.detail.startswith("готов за ")
