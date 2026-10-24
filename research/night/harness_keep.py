"""Харнесс QA, но папки прогонов (agent.log, llm_calls.jsonl, ответы) не удаляются — для разбора причин.

    uv run python research/night/harness_keep.py runs/night/harness5 --all --minutes 5
    uv run python research/night/harness_keep.py runs/llm15 --task tasks/public/regression/task.md --minutes 15 --llm

evaluation/harness.py работает во временной папке и стирает её вместе с agent.log; здесь временная папка
подменяется постоянной (первый аргумент). Остальные аргументы — как у evaluation.harness.

Как и CLI evaluation/harness.py, подхватывает .env из корня (ключ Qwen для --llm): без этого в agent.log
«LLM выключена…» и прогон с --llm молча идёт на playbook (#276). С --llm кэш ответов LLM_CACHE_DIR выключается —
на платформе кэша нет, а ответы из него приходят мгновенно и занижают время и расход; --cache оставляет кэш.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import harness  # noqa: E402


def main(argv: list[str]) -> int:
    """argv: папка для прогонов, затем аргументы evaluation.harness (+ --cache — не выключать кэш LLM)."""
    if not argv:
        print("Укажи папку для прогонов первым аргументом, например runs/night/harness5")
        return 2
    keep = Path(argv[0]).resolve()
    keep.mkdir(parents=True, exist_ok=True)
    rest = [a for a in argv[1:] if a != "--cache"]
    harness.load_env_file()  # не перезаписывает переменные, заданные в оболочке
    if "--llm" in rest:
        if not os.getenv("OPENAI_API_KEY"):
            print("--llm: нет OPENAI_API_KEY (ни в окружении, ни в .env) — агент пойдёт без Qwen")
            return 2
        if "--cache" not in argv:
            os.environ["LLM_CACHE_DIR"] = ""  # пустое значение уходит в дочерний процесс агента и выключает кэш

    @contextlib.contextmanager
    def _keep_dir(*args, **kwargs):
        yield str(keep)

    with mock.patch.object(harness.tempfile, "TemporaryDirectory", _keep_dir):
        return harness.main(rest)


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
