#!/usr/bin/env bash
# Ночной аудит: варианты раскладки/формата файлов задачи → что сделает solution.py (60 с, без LLM).
#   bash research/night/format_probe.sh [папка_для_вариантов]
# База — синтетическая t07 (runs/night/tasks, генератор research/night/gen_tasks.py).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
B="$ROOT/runs/night/tasks/t07_50classes_accuracy"
OUT="${1:-$ROOT/runs/night/fmt}"
mkdir -p "$OUT"
mk() { mkdir -p "$OUT/$1"; }
mk cp1251_md; cp "$B"/*.csv "$OUT/cp1251_md/"; iconv -f utf-8 -t cp1251 "$B/task.md" > "$OUT/cp1251_md/task.md"
mk tsv; cp "$B/task.md" "$OUT/tsv/"; for f in train test sample_submission; do sed 's/,/\t/g' "$B/$f.csv" > "$OUT/tsv/$f.tsv"; done
mk upper; cp "$B/task.md" "$OUT/upper/"; cp "$B/train.csv" "$OUT/upper/TRAIN.CSV"; cp "$B/test.csv" "$OUT/upper/Test.csv"; cp "$B/sample_submission.csv" "$OUT/upper/"
mk names; cp "$B/task.md" "$OUT/names/"; cp "$B/train.csv" "$OUT/names/data_train.csv"; cp "$B/test.csv" "$OUT/names/data_test.csv"; cp "$B/sample_submission.csv" "$OUT/names/submission_example.csv"
mk parquet; cp "$B/task.md" "$B/sample_submission.csv" "$OUT/parquet/"
"$PY" -c "import pandas as pd,sys; [pd.read_csv(f'$B/{f}.csv').to_parquet(f'$OUT/parquet/{f}.parquet') for f in ('train','test')]"
for t in cp1251_md tsv upper names parquet; do
  (cd "$OUT/$t" && LLM_MOCK=1 AGENT_TIME_BUDGET_SEC=60 timeout 120 "$PY" "$ROOT/solution.py" task.md > run.log 2>&1)
  if [ -f "$OUT/$t/submission.csv" ]; then echo "$t: submission.csv есть: $(sed -n 2p "$OUT/$t/submission.csv")"; else echo "$t: submission.csv НЕТ"; fi
done
