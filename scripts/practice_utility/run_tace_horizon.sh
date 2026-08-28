#!/usr/bin/env bash
# Stage 5: 128-iteration horizon cell, 4 arms x 3 seeds, after stage 4 finishes.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=11000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260818_141446/model_step_000024.pt
OUT="$LUCID_ROOT/outputs"
LOG=$OUT/tace_pilot_driver.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "stage 4 done" "$LOG"; do sleep 60; done
say "stage 5: horizon 128 iters, lucid fixed ta_lucid_25 ta_lucid_50 x 3 seeds"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs 128 --iterations 128 --warmup-iterations 10 \
  --seeds 8600 8601 8602 --modes lucid fixed ta_lucid_25 ta_lucid_50 \
  --min-free-mib "$MIN_FREE" --execute > "$OUT/tace_horizon_stdout.log" 2>&1
code=$?
receipt=$(grep -o "receipt .*json" "$OUT/tace_horizon_stdout.log" | tail -1 | cut -d' ' -f2)
say "stage 5 training exit=$code receipt=$receipt"
[ -z "$receipt" ] && { say "STAGE 5 FAILED"; exit 1; }
echo "$receipt" > "$OUT/tace_horizon.receipt"
python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$receipt" --num-envs 128 --seeds 8600 8601 8602 \
  --presets id_clean dr_050 dr_full latency_60ms --min-free-mib "$MIN_FREE" --execute \
  > "$OUT/tace_horizon_eval_stdout.log" 2>&1
say "stage 5 eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/tace_horizon_eval_stdout.log" | tail -1)"
say "=== stage 5 done ==="
