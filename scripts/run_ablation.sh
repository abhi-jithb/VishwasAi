#!/usr/bin/env bash
# Run A (retrieval ON) then B (retrieval OFF), cold cache, Ollama qwen2.5:7b. Logs to results/logs/.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-venv/bin/python}
export LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b PYTHONUNBUFFERED=1
mkdir -p results/logs
TS=$(date +%Y%m%d_%H%M%S)
for RUN in A B; do
  EXTRA=""; [ "$RUN" = B ] && EXTRA="--no-retrieval"
  LOG=results/logs/run_${RUN}_${TS}.log
  START=$(date +%s)
  echo "[$(date -Is)] start run $RUN" | tee -a results/logs/ablation_${TS}.txt
  $PY tests/run_gold.py --cold --model qwen2.5:7b --provider ollama $EXTRA > "$LOG" 2>&1
  CODE=$?
  END=$(date +%s)
  echo "[$(date -Is)] run $RUN exit=$CODE wall_clock_s=$((END-START)) log=$LOG $(grep -o 'saved to: .*' "$LOG")" | tee -a results/logs/ablation_${TS}.txt
done
