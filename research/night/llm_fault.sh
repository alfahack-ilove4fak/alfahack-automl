#!/usr/bin/env bash
# Ночной аудит: агент с поддельной LLM в режиме $1 на задаче $2 (папка), бюджет $3 с.
#   bash research/night/llm_fault.sh garbage tasks/practice/cancer_binary_auc 420
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd); PY="$ROOT/.venv/bin/python"
MODE=$1; TASK=$2; BUDGET=${3:-420}; PORT=$((8700 + RANDOM % 200))
OUT="$ROOT/runs/night/p4/llm_$MODE"; mkdir -p "$OUT"; rm -rf "${OUT:?}/task"; cp -r "$ROOT/$TASK" "$OUT/task"
"$PY" "$ROOT/research/night/fake_llm_marenkov.py" --port $PORT --mode "$MODE" --delay 400 > "$OUT/fake_llm.log" 2>&1 &
SRV=$!; sleep 1
start=$(date +%s)
(cd "$OUT/task" && env -u LLM_MOCK OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1 OPENAI_API_KEY=fake OPENAI_MODEL=fake \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost AGENT_TIME_BUDGET_SEC=$BUDGET AGENT_RUN_DIR="$OUT" \
  timeout $((BUDGET + 120)) "$PY" "$ROOT/solution.py" task.md > /dev/null 2>&1); code=$?
kill $SRV
echo "$MODE: exit=$code, $(( $(date +%s) - start )) с, submission: $([ -f "$OUT/task/submission.csv" ] && echo есть || echo НЕТ), запросов к LLM: $(wc -l < "$OUT/fake_llm.log")"
