#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
INITIAL="$ROOT/runs/mac-pretrain-before-after-001/checkpoint-0100"
SCREEN="$ROOT/runs/rl-variant-screen-001"
OUT="$ROOT/runs/rl-group-confirmation-001"
FAMILIES="increment,double,square,even,reverse"

mkdir -p "$OUT/evaluations"

train_arm() {
  local label="$1" seed="$2" groups="$3" group_size="$4"
  local run="$OUT/$label-seed-$seed"
  if [[ ! -f "$run/training-receipt.json" ]]; then
    "$PY" "$ROOT/scripts/train_rl.py" \
      --initial-checkpoint "$INITIAL" \
      --output "$run" \
      --groups "$groups" \
      --checkpoints "$groups" \
      --group-size "$group_size" \
      --max-new-tokens 32 \
      --temperature 0.35 \
      --learning-rate 5e-5 \
      --train-scope last-block-head \
      --reward-mode binary \
      --invalid-penalty -0.1 \
      --exact-bonus 0 \
      --families "$FAMILIES" \
      --tasks-per-family 8 \
      --seed "$seed" \
      --device mps
  fi
}

evaluate_arm() {
  local label="$1" seed="$2" groups="$3" checkpoint="$4"
  local evaluation="$OUT/evaluations/$label-seed-$seed-confirm.json"
  if [[ ! -f "$evaluation" ]]; then
    "$PY" "$ROOT/scripts/evaluate.py" \
      --checkpoint "$checkpoint" \
      --split confirm \
      --per-family 8 \
      --max-new-tokens 32 \
      --families "$FAMILIES" \
      --seed 20260827 \
      --device mps \
      --output "$evaluation"
  fi
}

if [[ ! -f "$OUT/evaluations/pretrained-confirm.json" ]]; then
  "$PY" "$ROOT/scripts/evaluate.py" \
    --checkpoint "$INITIAL" \
    --split confirm \
    --per-family 8 \
    --max-new-tokens 32 \
    --families "$FAMILIES" \
    --seed 20260827 \
    --device mps \
    --output "$OUT/evaluations/pretrained-confirm.json"
fi

# Seed 31415 was the screening run. The confirmation split was kept closed.
evaluate_arm baseline 31415 16 "$SCREEN/baseline/checkpoint-0016"
evaluate_arm group-4 31415 32 "$SCREEN/group-4/checkpoint-0032"

# Two fresh optimization/sampling seeds, with equal 128-rollout budgets.
for seed in 27182 16180; do
  train_arm baseline "$seed" 16 8
  train_arm group-4 "$seed" 32 4
  evaluate_arm baseline "$seed" 16 "$OUT/baseline-seed-$seed/checkpoint-0016"
  evaluate_arm group-4 "$seed" 32 "$OUT/group-4-seed-$seed/checkpoint-0032"
done

"$PY" "$ROOT/scripts/analyze_rl_group_confirmation.py" \
  --run-dir "$OUT" \
  --output "$OUT/CONFIRMATION-REPORT.md"
