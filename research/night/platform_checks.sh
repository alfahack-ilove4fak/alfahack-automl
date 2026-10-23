#!/usr/bin/env bash
# Ночной аудит, фаза 4: условия платформы. Запуск из корня репозитория:
#   bash research/night/platform_checks.sh 2>&1 | tee runs/night/platform_checks.txt
# 1) ZIP → чистая папка вне репозитория, только solution.py + requirements.txt + данные, без сети (unshare -n),
#    пустое окружение; 2) strace: куда агент ходит по сети; 3) kill -9 посреди цикла; 4) бюджет 30 и 120 с;
# 5) лимит памяти cgroup на задаче в 1 млн строк.
set -u
ROOT=$(pwd)
PY=$ROOT/.venv/bin/python
OUT=$ROOT/runs/night/platform
rm -rf "$OUT"; mkdir -p "$OUT"
TASK_SRC=$ROOT/runs/night/alien/ordinal_kappa_words   # «чужая» задача: в репозитории её нет

check() {  # check <папка задачи> — валиден ли submission.csv по ответам генератора
  "$PY" - "$1" <<'EOF'
import sys, json, pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
d = Path(sys.argv[1]); name = "ordinal_kappa_words"
ans = pd.read_csv(f"runs/night/alien/_answers/{name}.csv")
p = d / "submission.csv"
if not p.exists():
    print("   сабмит: НЕТ ФАЙЛА"); sys.exit()
sub = pd.read_csv(p)
ok = list(sub.columns) == list(ans.columns) and len(sub) == len(ans) and (sub.iloc[:, 0].astype(str) == ans.iloc[:, 0].astype(str)).all()
print(f"   сабмит: колонки/строки/id {'OK' if ok else 'НЕ OK'}; QWK={cohen_kappa_score(ans.iloc[:,1], sub.iloc[:,1].round().astype(int), weights='quadratic'):.4f}"
      f"; .tmp.csv остался: {(d / 'submission.tmp.csv').exists()}")
EOF
}

echo "== 1. Сборка ZIP"
uv run python scripts/package_submission.py 2>&1 | grep -v UV_NATIVE | tail -3
ZIP=$(ls -t dist/submission-*.zip | head -1)
echo "   архив: $ZIP, $(du -k "$ZIP" | cut -f1) КБ; внутри: $(unzip -Z1 "$ZIP" | tr '\n' ' ')"

echo "== 1b. Чистая папка вне репозитория, без сети, пустое окружение, бюджет 120 с"
CLEAN=$(mktemp -d /tmp/clean_platform_XXXX)
cp -r "$TASK_SRC"/. "$CLEAN"/
unzip -q -o "$ZIP" -d "$CLEAN"
ls "$CLEAN"
start=$(date +%s)
( cd "$CLEAN" && unshare -n env -i PATH=/usr/bin:/bin HOME=/tmp AGENT_TIME_BUDGET_SEC=120 \
    "$PY" solution.py task.md > "$OUT/clean_stdout.txt" 2>&1 ); code=$?
echo "   код выхода $code, $(( $(date +%s) - start )) с"
grep -E "Шаг 1|Сохранён ответ|Готово|ERROR|Traceback|Распаков|распаков" "$OUT/clean_stdout.txt" | cut -c1-200 | head -12
check "$CLEAN"

echo "== 2. strace: сетевые вызовы агента (LLM выключена), бюджет 60 с"
ST=$(mktemp -d /tmp/strace_XXXX); cp -r "$TASK_SRC"/. "$ST"/
( cd "$ST" && AGENT_TIME_BUDGET_SEC=60 LLM_MOCK=1 strace -f -qq -e trace=connect,sendto -o "$OUT/strace.txt" \
    "$PY" "$ROOT/solution.py" task.md > /dev/null 2>&1 )
echo "   connect к сети (не unix-сокеты): $(grep connect "$OUT/strace.txt" | grep -vc 'AF_UNIX\|AF_LOCAL')"
grep connect "$OUT/strace.txt" | grep -v 'AF_UNIX\|AF_LOCAL' | head -5

echo "== 3. kill -9 посреди цикла (бюджет 300 с, убиваем на 40-й секунде)"
K=$(mktemp -d /tmp/kill_XXXX); cp -r "$TASK_SRC"/. "$K"/
( cd "$K" && AGENT_TIME_BUDGET_SEC=300 LLM_MOCK=1 AGENT_RUN_DIR="$K/run" "$PY" "$ROOT/solution.py" task.md > /dev/null 2>&1 & echo $! > "$K/pid" )
sleep 40; kill -9 "$(cat "$K/pid")"; sleep 1
echo "   до убийства в логе: $(grep -c 'Итерация' "$K/run/agent.log") итераций, последняя запись ответа: $(grep -o 'Сохранён ответ «[^»]*»' "$K/run/agent.log" | tail -1)"
check "$K"

echo "== 3b. SIGTERM посреди цикла (как остановка платформы)"
K2=$(mktemp -d /tmp/term_XXXX); cp -r "$TASK_SRC"/. "$K2"/
( cd "$K2" && AGENT_TIME_BUDGET_SEC=300 LLM_MOCK=1 AGENT_RUN_DIR="$K2/run" "$PY" "$ROOT/solution.py" task.md > /dev/null 2>&1 & echo $! > "$K2/pid" )
sleep 40; t0=$(date +%s); kill -TERM "$(cat "$K2/pid")"
while kill -0 "$(cat "$K2/pid")" 2>/dev/null; do sleep 0.2; done
echo "   процесс завершился через $(( $(date +%s) - t0 )) с после SIGTERM"
grep -E "сигнал|Финал" "$K2/run/agent.log" | cut -c1-160 | tail -3
check "$K2"

for B in 30 120; do
  echo "== 4. Бюджет AGENT_TIME_BUDGET_SEC=$B на public/multiclass (87 тыс. строк)"
  S=$(mktemp -d /tmp/budget_XXXX); cp -r "$ROOT/tasks/public/multiclass"/. "$S"/; rm -f "$S/submission.csv"
  start=$(date +%s)
  ( cd "$S" && AGENT_TIME_BUDGET_SEC=$B LLM_MOCK=1 AGENT_RUN_DIR="$S/run" "$PY" "$ROOT/solution.py" task.md > /dev/null 2>&1 )
  echo "   завершился за $(( $(date +%s) - start )) с (бюджет $B с)"
  grep -E "Шаг 0|Шаг 1|Сохранён ответ|сторож|Сторож|Финал|бейзлайн «" "$S/run/agent.log" | cut -c1-170 | head -8
  "$PY" -c "
import pandas as pd, sys
s = pd.read_csv('$S/submission.csv'); print('   сабмит:', s.shape, 'классов в ответе:', s.iloc[:, -1].nunique())"
done

echo "== 5. Лимит памяти cgroup 1500 МБ, задача 1 млн строк, бюджет 300 с"
M=$(mktemp -d /tmp/mem_XXXX); cp -r "$ROOT/runs/night/alien/million_rows_rmsle"/. "$M"/
CG=/sys/fs/cgroup/memory/agent_audit; mkdir -p $CG; echo $((1500*1024*1024)) > $CG/memory.limit_in_bytes
echo 0 > $CG/memory.swappiness 2>/dev/null
start=$(date +%s)
( cd "$M" && sh -c "echo \$\$ > $CG/cgroup.procs; exec env AGENT_TIME_BUDGET_SEC=300 LLM_MOCK=1 AGENT_RUN_DIR=$M/run $PY $ROOT/solution.py task.md" > "$OUT/mem_stdout.txt" 2>&1 ); code=$?
echo "   код выхода $code (137 = убит OOM), $(( $(date +%s) - start )) с; OOM-убийств: $(cat $CG/memory.oom_control | grep oom_kill | awk '{print $2}')"
grep -E "RAM|Шаг 0|Шаг 1|Сохранён ответ|подвыборке|память|memory|Финал" "$M/run/agent.log" | cut -c1-170 | head -8
ls -la "$M/submission.csv" 2>&1 | cut -c1-120
rmdir $CG 2>/dev/null
echo "== готово"
