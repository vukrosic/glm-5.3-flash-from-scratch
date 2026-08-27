#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
INITIAL="$ROOT/runs/mac-pretrain-before-after-001/checkpoint-0100"
OUT="$ROOT/runs/rl-variant-screen-001"
FAMILIES="increment,double,square,even,reverse"

mkdir -p "$OUT/evaluations"

run_arm() {
  local name="$1" groups="$2" group_size="$3" reward="$4" penalty="$5" temperature="$6"
  local run="$OUT/$name"
  if [[ ! -f "$run/training-receipt.json" ]]; then
    "$PY" "$ROOT/scripts/train_rl.py" \
      --initial-checkpoint "$INITIAL" \
      --output "$run" \
      --groups "$groups" \
      --checkpoints "$groups" \
      --group-size "$group_size" \
      --max-new-tokens 32 \
      --temperature "$temperature" \
      --learning-rate 5e-5 \
      --train-scope last-block-head \
      --reward-mode "$reward" \
      --invalid-penalty "$penalty" \
      --exact-bonus 0 \
      --families "$FAMILIES" \
      --tasks-per-family 8 \
      --seed 31415 \
      --device mps
  fi
  local checkpoint
  checkpoint=$(printf '%s/checkpoint-%04d' "$run" "$groups")
  local evaluation="$OUT/evaluations/$name-dev.json"
  if [[ ! -f "$evaluation" ]]; then
    "$PY" "$ROOT/scripts/evaluate.py" \
      --checkpoint "$checkpoint" \
      --split dev \
      --per-family 4 \
      --max-new-tokens 32 \
      --families "$FAMILIES" \
      --seed 2026 \
      --device mps \
      --output "$evaluation"
  fi
}

if [[ ! -f "$OUT/evaluations/pretrained-dev.json" ]]; then
  "$PY" "$ROOT/scripts/evaluate.py" \
    --checkpoint "$INITIAL" \
    --split dev \
    --per-family 4 \
    --max-new-tokens 32 \
    --families "$FAMILIES" \
    --seed 2026 \
    --device mps \
    --output "$OUT/evaluations/pretrained-dev.json"
fi

# All arms use exactly 128 sampled training completions.
run_arm baseline     16 8  binary        -0.1 0.35
run_arm partial      16 8  case-fraction -0.1 0.35
run_arm invalid-zero 16 8  binary         0.0 0.35
run_arm invalid-hard 16 8  binary        -1.0 0.35
run_arm temp-low     16 8  binary        -0.1 0.20
run_arm temp-high    16 8  binary        -0.1 0.80
run_arm group-4      32 4  binary        -0.1 0.35
run_arm group-16      8 16 binary        -0.1 0.35

"$PY" "$ROOT/scripts/analyze_rl_variants.py" \
  --screen-dir "$OUT" \
  --output "$OUT/SCREENING-REPORT.md"
