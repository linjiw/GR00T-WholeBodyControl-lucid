#!/usr/bin/env bash
# Re-run stage 8's evaluation, which died 54 cells into 75.
#
# WHAT WENT WRONG. Three GPU jobs were in flight at once: stage 8's evaluation,
# a released-checkpoint evaluation I started alongside it to "save time", and
# the from-scratch pilot, which woke at 12:44 when the card briefly showed
# 14.3 GB free. At 12:54 stage 8's next arm found 4,580 MiB free against a
# 9,000 MiB gate and raised immediately -- because `run_arm` reads
# LUCID_GPU_WAIT_SECONDS and it defaults to **0**, so the gate is a kill switch
# rather than a queue unless a driver sets it. None of my drivers set it.
#
# Two fixes, both here. Serialize: wait for the pilot and the release evaluation
# to finish before starting. And set LUCID_GPU_WAIT_SECONDS, so that if anything
# else does take the card this queues behind it instead of dying 72% of the way
# through a preregistered primary.
#
# The 54 completed cells are not reusable: the evaluator has no cell-level
# resume and the crashed run wrote no receipt, so the 21 missing cells exist on
# disk with nothing indexing them. Re-running all 75 is the cheap, honest option
# at ~2.4 h; splitting the receipt in two to save 50 cells would leave the
# preregistered primary spread across two artifacts for no scientific gain.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"

# Queue behind other jobs rather than die at the first arm.
export LUCID_GPU_WAIT_SECONDS=43200

OUT="$LUCID_ROOT/outputs"
LOG="$OUT/lucid_s_driver.log"
S8=$(grep -o "receipt [^ ]*json" "$OUT/stage8_lucids_stdout.log" | tail -1 | cut -d' ' -f2)
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "stage 8 eval recovery armed; waiting for the pilot and the release evaluation"
while pgrep -f "practice_utility/run_scratch_pilot.sh" >/dev/null \
   || pgrep -f "release_local_pseudo_receipt" >/dev/null; do sleep 120; done
say "card is free; re-running stage 8 evaluation in full (receipt $S8)"

python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$S8" --num-envs 128 --seeds 8600 8601 8602 \
  --presets id_clean dr_050 dr_full dr_125 latency_60ms \
  --smpl-motion-file dummy --min-free-mib 9000 --execute \
  > "$OUT/eval_stage8_retry_stdout.log" 2>&1
say "stage 8 eval retry exit=$? receipt=$(grep -o 'receipt [^ ]*json' "$OUT/eval_stage8_retry_stdout.log" | tail -1 | cut -d' ' -f2)"
say "=== stage 8 evaluation recovered ==="
