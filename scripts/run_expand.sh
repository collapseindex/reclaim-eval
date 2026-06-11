#!/usr/bin/env bash
# Core expansion runs, in priority order. Checkpointed + resumable: re-running skips
# finished units, so a hang or restart costs nothing already paid for.
set -u
cd "$(dirname "$0")/.."
LOG=data/expand.log
GROK=x-ai/grok-4.3-20260430
LLAMA=meta-llama/llama-3.1-8b-instruct

run () {  # label model task extra-args...
  local label=$1 model=$2 task=$3; shift 3
  echo "=== START $label ===" >> "$LOG"
  python -u scripts/run_pilot.py --real --fix --task "$task" --model "$model" \
    --seeds 3 --temp 0.7 "$@" >> "$LOG" 2>&1
  echo "=== DONE $label rc=$? ===" >> "$LOG"
}

# reuse the already-paid grok arith data (lossy + source_first) under the task-tagged
# name, so the arith control run only has to add the lossy_padded policy.
OLD=data/results/fix_x-ai_grok-4.3-20260430.jsonl
NEW=data/results/fix_x-ai_grok-4.3-20260430_arith.jsonl
if [ -f "$OLD" ] && [ ! -f "$NEW" ]; then mv "$OLD" "$NEW"; echo "reused grok arith ckpt" >> "$LOG"; fi

# 1) the falsifier: non-arith logic on the frontier model (the key result)
run grok-logic   "$GROK"  logic
# 2) arith length-control on the frontier model (adds lossy_padded to existing data)
run grok-arith   "$GROK"  arith
# 3) second model on both (pennies; pass llama prices for honest cost lines)
run llama-logic  "$LLAMA" logic  --price-in 0.02 --price-out 0.03
run llama-arith  "$LLAMA" arith  --price-in 0.02 --price-out 0.03

echo "=== ALLDONE ===" >> "$LOG"
