"""Сторож: дедлайн, сигнал остановки, блокировка записи, устойчивость к ошибкам финализации."""

import os
import signal
import threading
import time

import pytest

from runtime.watchdog import Watchdog


class Recorder:
    def __init__(self):
        self.reasons: list[str] = []
        self.exit_codes: list[int] = []
        self.finalized = threading.Event()
        self.exited = threading.Event()

    def finalize(self, reason: str) -> None:
        self.reasons.append(reason)
        self.finalized.set()

    def exit(self, code: int) -> None:
        self.exit_codes.append(code)
        self.exited.set()


def test_fires_at_deadline():
    rec = Recorder()
    dog = Watchdog(time.time() + 0.3, rec.finalize, exit_fn=rec.exit)
    dog.start(handle_signals=False)
    assert rec.exited.wait(5)
    assert rec.reasons == ["жёсткий дедлайн"]
    assert rec.exit_codes == [0]
    dog.stop()


def test_stop_cancels_deadline():
    rec = Recorder()
    dog = Watchdog(time.time() + 0.5, rec.finalize, exit_fn=rec.exit)
    dog.start(handle_signals=False)
    dog.stop()
    time.sleep(0.8)
    assert not rec.exited.is_set()


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="нужны POSIX-сигналы")
def test_sigterm_triggers_finalize_and_exit():
    rec = Recorder()
    previous = signal.getsignal(signal.SIGTERM)
    dog = Watchdog(time.time() + 60, rec.finalize, exit_fn=rec.exit)
    dog.start()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        assert rec.exited.wait(5)
        assert rec.reasons == ["сигнал SIGTERM"]
        assert rec.exit_codes == [128 + signal.SIGTERM]
    finally:
        dog.stop()
    assert signal.getsignal(signal.SIGTERM) == previous  # обработчики вернулись на место


def test_waits_for_answer_write_before_finalizing():
    rec = Recorder()
    dog = Watchdog(time.time() + 0.2, rec.finalize, exit_fn=rec.exit, lock_timeout=5)
    dog.start(handle_signals=False)
    with dog.critical():  # «пишем ответ» дольше дедлайна
        time.sleep(0.6)
        assert not rec.finalized.is_set()
    assert rec.exited.wait(5)
    assert rec.reasons == ["жёсткий дедлайн"]


def test_exits_even_if_write_never_finishes():
    rec = Recorder()
    dog = Watchdog(time.time() + 0.1, rec.finalize, exit_fn=rec.exit, lock_timeout=0.3)
    with dog.critical():  # лок так и не отпускаем
        dog.start(handle_signals=False)
        assert rec.exited.wait(5)


def test_exits_if_finalize_raises_or_hangs():
    def broken(reason: str) -> None:
        raise RuntimeError("boom")

    rec = Recorder()
    dog = Watchdog(time.time() + 0.1, broken, exit_fn=rec.exit)
    dog.start(handle_signals=False)
    assert rec.exited.wait(5)

    rec2 = Recorder()
    dog2 = Watchdog(time.time() + 0.1, lambda reason: time.sleep(30), exit_fn=rec2.exit, finalize_timeout=0.3)
    dog2.start(handle_signals=False)
    assert rec2.exited.wait(5)
