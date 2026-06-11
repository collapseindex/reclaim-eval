#!/usr/bin/env bash
# Relaunch after hardening the client + fixing the logic `ask`. Resumable: skips units
# already in the checkpoints. Cheap valid llama shapes first; grok-logic stays parked.
set -u
cd "$(dirname "$0")/.."
LOG=data/expand2.log
LLAMA=meta-llama/llama-3.1-8b-instruct
GROK=x-ai/grok-4.3-20260430

run () {  # label model task extra-args...
  local label=$1 model=$2 task=$3; shift 3
  echo "=== START $label ===" >> "$LOG"
  python -u scripts/run_pilot.py --real --fix --task "$task" --model "$model" \
    --seeds 3 --temp 0.7 "$@" >> "$LOG" 2>&1
  echo "=== DONE $label rc=$? ===" >> "$LOG"
}

run llama-logic "$LLAMA" logic --price-in 0.02 --price-out 0.03
run llama-arith "$LLAMA" arith --price-in 0.02 --price-out 0.03
run grok-arith  "$GROK"  arith   # resume: only the remaining lossy_padded control units

echo "=== ALLDONE ===" >> "$LOG"
