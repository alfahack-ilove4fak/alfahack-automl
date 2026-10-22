"""Харнесс QA, но папки прогонов (agent.log, llm_calls.jsonl, ответы) не удаляются — для разбора причин.

    uv run python research/night/harness_keep.py runs/night/harness5 --all --minutes 5

evaluation/harness.py работает во временной папке и стирает её вместе с agent.log; здесь временная папка
подменяется постоянной (первый аргумент). Остальные аргументы — как у evaluation.harness.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import harness  # noqa: E402

keep = Path(sys.argv.pop(1)).resolve()
keep.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _keep_dir(*args, **kwargs):
    yield str(keep)


harness.tempfile.TemporaryDirectory = _keep_dir

if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(harness.main())
