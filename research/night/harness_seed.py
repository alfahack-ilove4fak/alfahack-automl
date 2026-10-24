"""harness_keep с другим SEED разбиения — независимая отложенная часть (#276).

    uv run python research/night/harness_seed.py 7 runs/mc7 --task tasks/public/multiclass/task.md --minutes 15 --agent-root <worktree>

Первый аргумент — seed, остальные — как у research/night/harness_keep.py.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import harness_keep  # noqa: E402

from evaluation import harness  # noqa: E402

if __name__ == "__main__":
    harness.SEED = int(sys.argv[1])
    sys.exit(harness_keep.main(sys.argv[2:]))
