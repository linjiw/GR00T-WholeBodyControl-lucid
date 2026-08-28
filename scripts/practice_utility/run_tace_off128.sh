#!/usr/bin/env bash
# Stage 6: no-DR control at 128 iterations (drift vs DR-induced degradation).
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
until grep -q "stage 5 done" "$LOG"; do sleep 60; done
say "stage 6: off x 3 seeds x 128 iters (drift control)"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs 128 --iterations 128 --warmup-iterations 10 \
  --seeds 8600 8601 8602 --modes off --min-free-mib "$MIN_FREE" --execute > "$OUT/tace_off128_stdout.log" 2>&1
code=$?
receipt=$(grep -o "receipt .*json" "$OUT/tace_off128_stdout.log" | tail -1 | cut -d' ' -f2)
say "stage 6 training exit=$code receipt=$receipt"
[ -z "$receipt" ] && { say "STAGE 6 FAILED"; exit 1; }
echo "$receipt" > "$OUT/tace_off128.receipt"
python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$receipt" --num-envs 128 --seeds 8600 8601 8602 \
  --presets id_clean dr_050 dr_full latency_60ms --min-free-mib "$MIN_FREE" --execute \
  > "$OUT/tace_off128_eval_stdout.log" 2>&1
say "stage 6 eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/tace_off128_eval_stdout.log" | tail -1)"
say "=== stage 6 done ==="
