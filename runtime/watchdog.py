"""Сторож последней минуты: жёсткая отсечка по времени и обработка сигналов остановки. Владелец — капитан.

Зачем отдельный поток. LightGBM и другие C-библиотеки не отдают управление питону минутами, поэтому
обычный `signal.signal`-обработчик в главном потоке сработал бы только после окончания обучения, а проверки
`budget.has_time()` — только между фолдами. Сторож живёт в своём потоке (C-код GIL не держит) и ждёт одного из двух:
  1. наступил жёсткий дедлайн (`hard_deadline`, unix-время);
  2. пришёл SIGTERM / SIGINT / SIGHUP — сигнал будит сторожа через `signal.set_wakeup_fd`,
     то есть сразу, а не когда главный поток вернётся из C-кода.
Тогда сторож берёт блокировку записи ответа (чтобы не оборвать запись посередине), вызывает `finalize`
(проверить submission.csv и при необходимости вернуть страховочный ответ) и завершает процесс через `os._exit`.

Гарантия «submission.csv валиден» держится не на сторожe, а на атомарной записи (`write_submission_atomic`):
сторож лишь добавляет, что процесс не зависнет и не будет убит посреди записи.
"""

from __future__ import annotations

import contextlib
import logging
import os
import select
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterator

from runtime.log import get_logger

log = get_logger("watchdog")

_STOP_BYTE = b"\x00"
_POLL_SECONDS = 5.0  # как часто сверять часы, если сигналов нет


class Watchdog:
    def __init__(
        self,
        hard_deadline: float,
        finalize: Callable[[str], None],
        *,
        lock: threading.Lock | None = None,
        exit_fn: Callable[[int], None] = os._exit,
        lock_timeout: float = 15.0,
        finalize_timeout: float = 30.0,
    ) -> None:
        self.hard_deadline = hard_deadline
        self._finalize = finalize
        self._lock = lock or threading.Lock()
        self._exit = exit_fn
        self._lock_timeout = lock_timeout
        self._finalize_timeout = finalize_timeout
        self._thread: threading.Thread | None = None
        self._rsock: socket.socket | None = None
        self._wsock: socket.socket | None = None
        self._old_wakeup_fd: int | None = None
        self._old_handlers: dict[int, object] = {}
        self._stopped = threading.Event()
        self.fired = threading.Event()
        self.reason: str | None = None

    @contextlib.contextmanager
    def critical(self) -> Iterator[None]:
        """Участок, который нельзя обрывать (запись ответа): сторож подождёт его конца, прежде чем выйти."""
        with self._lock:
            yield

    def start(self, handle_signals: bool = True) -> None:
        if handle_signals:
            self._install_signals()
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Штатное завершение: убрать сторожа и вернуть прежние обработчики сигналов."""
        self._stopped.set()
        if self._wsock is not None:
            with contextlib.suppress(OSError):
                self._wsock.send(_STOP_BYTE)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self._restore_signals()

    # --- сигналы -------------------------------------------------------------------------------------

    def _install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            log.warning("Сторож запущен не в главном потоке — сигналы не перехватываются, работает только дедлайн")
            return
        try:
            self._rsock, self._wsock = socket.socketpair()
            self._rsock.setblocking(False)
            self._wsock.setblocking(False)
            self._old_wakeup_fd = signal.set_wakeup_fd(self._wsock.fileno(), warn_on_full_buffer=False)
            for name in ("SIGTERM", "SIGINT", "SIGHUP"):
                sig = getattr(signal, name, None)
                if sig is not None:
                    # Обработчик-пустышка нужен, чтобы C-уровень писал номер сигнала в wakeup_fd
                    self._old_handlers[sig] = signal.signal(sig, lambda *_: None)
        except (OSError, ValueError):
            log.exception("Не удалось перехватить сигналы остановки — остаётся только дедлайн")
            self._restore_signals()

    def _restore_signals(self) -> None:
        for sig, handler in self._old_handlers.items():
            with contextlib.suppress(OSError, ValueError, TypeError):
                signal.signal(sig, handler)  # type: ignore[arg-type]
        self._old_handlers.clear()
        if self._old_wakeup_fd is not None:
            with contextlib.suppress(OSError, ValueError):
                signal.set_wakeup_fd(self._old_wakeup_fd)
            self._old_wakeup_fd = None
        for sock in (self._rsock, self._wsock):
            if sock is not None:
                sock.close()
        self._rsock = self._wsock = None

    # --- поток сторожа -------------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._watch()
        except Exception:  # noqa: BLE001 — упавший сторож не должен оставить процесс без реакции на Ctrl-C
            log.exception("Сторож упал — возвращаем обычные обработчики сигналов")
            self._restore_signals()

    def _watch(self) -> None:
        rsock = self._rsock
        while not self._stopped.is_set():
            left = self.hard_deadline - time.time()
            if left <= 0:
                self._fire("жёсткий дедлайн", exit_code=0)
                return
            timeout = min(left, _POLL_SECONDS)
            if rsock is None:
                self._stopped.wait(timeout)
                continue
            try:
                readable, _, _ = select.select([rsock], [], [], timeout)
                data = rsock.recv(64) if readable else b""
            except (OSError, ValueError):
                if self._stopped.is_set():
                    return
                self._stopped.wait(0.5)  # сокет сломан — не крутимся вхолостую, дедлайн всё равно сработает
                rsock = None
                continue
            for byte in data:
                if byte == 0:
                    continue  # штатная остановка
                try:
                    name = signal.Signals(byte).name
                except ValueError:
                    name = str(byte)
                self._fire(f"сигнал {name}", exit_code=128 + byte)
                return

    def _fire(self, reason: str, exit_code: int) -> None:
        if self.fired.is_set():
            return
        self.reason = reason
        self.fired.set()
        log.warning("Сторож сработал: %s — финализируем ответ и выходим", reason)
        got_lock = self._lock.acquire(timeout=self._lock_timeout)  # даём дописать ответ, если он пишется
        if not got_lock:
            log.error("Запись ответа не закончилась за %.0f с — выходим без ожидания", self._lock_timeout)
        worker = threading.Thread(target=self._finalize_safely, args=(reason,), name="finalize", daemon=True)
        worker.start()
        worker.join(self._finalize_timeout)
        if worker.is_alive():
            log.error("Финализация зависла (> %.0f с) — выходим как есть", self._finalize_timeout)
        for handler in logging.getLogger("agent").handlers:
            with contextlib.suppress(Exception):
                handler.flush()
        self._exit(exit_code)

    def _finalize_safely(self, reason: str) -> None:
        try:
            self._finalize(reason)
        except Exception:  # noqa: BLE001 — сторож не должен падать, иначе процесс так и останется висеть
            log.exception("Ошибка при финализации ответа")
