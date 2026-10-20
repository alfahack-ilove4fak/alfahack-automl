"""Единый лог агента: в stderr (видно на платформе) и в runs/<время>/agent.log (видно локально)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_STARTED = time.time()


class _ElapsedFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.elapsed = f"{time.time() - _STARTED:7.1f}s"
        return super().format(record)


def setup_logging(run_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("agent")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = _ElapsedFormatter("[%(elapsed)s] %(levelname)-5s %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(run_dir / "agent.log", encoding="utf-8")
        file.setFormatter(fmt)
        logger.addHandler(file)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"agent.{name}")
